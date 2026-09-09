// Name the next sensitive module from where the track already is, using only
// public geometry calls and no propagation.
//
// Why this exists. `Navigator::resolveSurfaces` returns an
// empty candidate list while `state.currentLayer` is null (`Navigator.cpp:669`),
// and `currentLayer` is set only on arrival at the layer's own
// `surfaceRepresentation` (`:398`), which is its approach surface. So inside the
// CKF a module becomes the stepper's target only for the last few millimetres
// of a leg. The three calls the navigator makes to get there are all public:
//
//   TrackingVolume::compatibleLayers      (TrackingVolume.hpp:407)
//   Layer::compatibleSurfaces             (Layer.hpp:170)
//   TrackingVolume::compatibleBoundaries  (TrackingVolume.hpp:423)
//
// One walk lives here rather than in each probe. `nav_probe` measures what the
// walk can name over the whole detector; `nav_truth` measures whether what it
// names is what a propagation reaches. Those two have to be the same walk or
// the second does not test the first, so neither owns it.
//
// All of it is geometry lookup. Nothing here steps, integrates or reads a
// field, so what the walk costs is not what a propagation costs and a timing
// taken on it would be quoted later as if it were.
//
// Three traps all return empty rather than failing: `startObject` is where the
// layer walk starts and not an object to skip; `compatibleLayers` has to be
// asked from a point inside
// the volume being asked; and one boundary crossing does not reach the next
// equipped volume, because the ODD interleaves `PST::Barrel` and the `*::fGap`
// and `*::sGap` volumes, which carry navigation layers and no sensitive
// surfaces.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Propagator/NavigationTarget.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/SurfaceArray.hpp"
#include "Acts/Utilities/Logger.hpp"

namespace nav_walk {

struct Options {
  /// A surface at zero path length is the one the query started on. The
  /// navigator excludes that through `startObject`; this also holds a floor,
  /// because a module the ray happens to graze is not a transport.
  double nearLimit = 1e-3;

  /// How far past a volume boundary to look up the volume beyond it.
  /// `TrackingGeometry::lowestTrackingVolume` resolves with
  /// `s_onSurfaceTolerance` (1e-4 mm), so a point exactly on the crossing can
  /// come back as the volume just left.
  double stepPast = 1e-2;

  /// The tracker. Everything outside it is solenoid, calorimeter or air, and no
  /// transport this project prices happens there. Same cut as
  /// `field_at_modules.py`.
  double rMax = 1080.0;
  double zMax = 3030.0;

  /// How many volume boundaries the walk will cross before giving up. The ODD
  /// puts up to two unequipped volumes between two equipped ones, so a bound
  /// below about four would report a geometry limit that is really this limit.
  std::size_t maxCrossings = 8;

  /// The boundary check `Layer::compatibleSurfaces` applies to each module it
  /// intersects. `Layer.cpp` hands `options.boundaryTolerance` straight to
  /// `surface.intersect`, and the navigator's own default is
  /// `BoundaryTolerance::None()`, an exact edge test. Relaxing it is one of the
  /// two things that can put a module back in the candidate list; the other is
  /// the binned lookup upstream of it, which this does not touch.
  ///
  /// Applied to the SURFACE query only. `compatibleLayers` keeps `None()`, so
  /// the layer nomination is the same in a relaxed pass as in an exact one and
  /// the two are comparable.
  Acts::BoundaryTolerance surfaceTolerance = Acts::BoundaryTolerance::None();
};

/// One sensitive surface the chosen layer offered, with its straight-line path
/// from the query point. The list is what a caller needs to ask whether the
/// module a propagation actually reached was nominated at all, and where in the
/// order it sat.
struct Candidate {
  const Acts::Surface* surface = nullptr;
  double path = 0.0;
};

/// Which of the four roles a surface plays, off its geometry identifier.
///
/// `sensitive()` non-zero is a module, `approach()` non-zero is one of a
/// layer's approach-descriptor surfaces, `boundary()` non-zero is a volume
/// boundary, and layer-scoped with none of those is the layer's own
/// representing surface. Counted this way the ODD has 123 material-carrying
/// surfaces: 10 approach descriptor, 89 layer representing, 24 volume
/// boundary, 0 sensitive.
enum class Kind : int {
  Module = 0,
  Approach = 1,
  Representing = 2,
  Boundary = 3,
  Other = 4,
};

inline Kind kindOf(const Acts::Surface& s) {
  const Acts::GeometryIdentifier id = s.geometryId();
  if (id.sensitive() != 0) {
    return Kind::Module;
  }
  if (id.approach() != 0) {
    return Kind::Approach;
  }
  if (id.boundary() != 0) {
    return Kind::Boundary;
  }
  if (id.layer() != 0) {
    return Kind::Representing;
  }
  return Kind::Other;
}

/// One material-carrying surface the leg crosses on the way to the named
/// module, with its straight-line path from the query point.
///
/// This is what a moved seam owes and a stepper seam does not. The stock
/// navigator stops the stepper at each of these and the CKF applies the
/// material there (`CombinatorialKalmanFilter.hpp:473-478`). A navigator that
/// names the far module does not stop, so nothing applies it, which costs 19.7
/// efficiency points in the softest pT bin. The three kinds come out of
/// different branches of the geometry and
/// none of them is a module: the approach and representing surfaces from
/// `Layer::compatibleSurfaces` sections (A) and (C), which this walk already
/// calls and used to discard, and the volume boundaries from
/// `TrackingVolume::compatibleBoundaries`, which it already calls to cross.
///
/// The path is the straight ray's, as everything else in this header is. It is
/// used to decide when a crossing is reached and not where: a caller applying
/// the material should re-intersect the surface from wherever its own
/// trajectory has got to, because over a 225 mm leg a 1 GeV helix leaves the
/// ray by about 4 mm.
struct Crossing {
  const Acts::Surface* surface = nullptr;
  double path = 0.0;
  Kind kind = Kind::Other;
};

/// What happened in the volume the query started in, kept apart from the rest
/// because the two failures mean opposite things: `NoLayer` is a track on the
/// outermost layer of its volume and is the geometry answering correctly,
/// `NoModule` is every layer of the volume having been asked and none of them
/// holding a module along this ray.
enum class FirstVolume { Resolved, NoLayer, NoModule };

struct Result {
  bool ok = false;
  /// Set when `ok` is false. One of the strings below.
  const char* stopped = "";
  FirstVolume firstVolume = FirstVolume::Resolved;

  /// Volume boundaries crossed before the module was named.
  std::size_t crossings = 0;
  /// Path from the query point to the crossing the module was named after.
  double crossingPath = 0.0;
  /// The crossings themselves, in order, for a caller that reports them.
  std::vector<std::pair<const Acts::TrackingVolume*, const Acts::TrackingVolume*>>
      crossed;

  /// The layer the module was found on, and the path to its approach surface.
  const Acts::Layer* layer = nullptr;
  double layerPath = 0.0;

  /// The approach surface itself, which is what `compatibleLayers` intersected
  /// to get `layerPath`. A caller that advances a curve to the layer needs it
  /// to re-intersect from wherever the curve landed.
  ///
  /// `TrackingVolume::compatibleLayers` builds its targets from
  /// `Layer::surfaceOnApproach`, so `NavigationTarget::surface()` on a layer
  /// target is the approach surface and not the module.
  const Acts::Surface* layerSurface = nullptr;

  /// The point the resolving `compatibleLayers`/`compatibleSurfaces` pair was
  /// actually asked from. It is the query point after the last volume crossing,
  /// not the caller's `p`, and the two differ by `crossingPath`. Anything that
  /// re-asks the geometry the same question -- `SurfaceArray::neighbors`, for
  /// instance -- has to ask from here or it is asking a different question.
  Acts::Vector3 queryPos = Acts::Vector3::Zero();

  /// Every layer of the resolving volume that lay ahead along the ray, in path
  /// order, whether or not it offered a module. `layer` is the first of these
  /// that did.
  std::vector<const Acts::Layer*> layersAhead;

  /// The straight-line nearest sensitive surface on that layer, which is the
  /// walk's own answer, and every sensitive surface that layer offered.
  const Acts::Surface* nearest = nullptr;
  double nearestPath = 0.0;
  std::vector<Candidate> candidates;

  /// What the layer the track is standing on still offers ahead.
  ///
  /// The walk names no module there, by design: excluding the start layer is
  /// what makes the 99.73 % a measurement of naming the module on the next
  /// layer, and 17,824 of 68,343 transitions have their successor on
  /// the start layer and are outside its domain.
  ///
  /// A caller that replaces the navigator cannot leave those 26 % unanswered.
  /// The stock navigator names them out of the candidate list it resolved when
  /// it entered the layer (`Navigator.cpp:436-450`), and a seam that skips to
  /// the next layer skips the measurement with it. This is the field that lets
  /// a caller see the case and decline the leg, which hands it back to the
  /// mechanism that has always named it.
  ///
  /// Empty when nothing on the start layer lies ahead, and empty when the walk
  /// was given no start layer.
  const Acts::Surface* startLayerNearest = nullptr;
  double startLayerPath = 0.0;
  std::vector<Candidate> startLayerCandidates;

  /// Every material-carrying non-sensitive surface between the query point and
  /// `nearest`, in path order, deduplicated, with the paths carried back to the
  /// caller's `p`.
  ///
  /// Empty when `ok` is false: a walk that named no module has no leg and the
  /// list would describe a distance the caller is not going to fly.
  ///
  /// Collected from three places the walk already visits and used to throw
  /// away. The fields above cannot produce this; the calls behind them can.
  std::vector<Crossing> material;
};

/// What one layer offers along one ray. The second stage of a two-stage
/// resolve, and the same call `Layer::compatibleSurfaces` makes in the walk,
/// lifted out so a caller can repeat it from a different point without
/// repeating the layer nomination.
struct LayerResolve {
  bool ok = false;
  const Acts::Surface* nearest = nullptr;
  double nearestPath = 0.0;
  std::vector<Candidate> candidates;

  /// This layer's own material-carrying non-sensitive surfaces along the ray,
  /// which are sections (A) and (C) of `Layer::compatibleSurfaces`. Filled
  /// whether or not `ok`: a layer that offers no module is still crossed, and
  /// its material is still on the leg.
  std::vector<Crossing> material;
};

/// Why a walk stopped. Compared by pointer in the callers, so they are the one
/// definition rather than string literals spread over two files.
inline constexpr const char* kStoppedExhausted = "crossings exhausted";
inline constexpr const char* kStoppedNoBoundary = "no boundary ahead";
inline constexpr const char* kStoppedNoVolume = "no volume beyond";
inline constexpr const char* kStoppedLeftTracker = "left the tracker";
inline constexpr const char* kStoppedNoStartVolume = "no volume at the start";

namespace detail {

/// Ask one layer for its sensitive surfaces, from `p`, along `d`.
///
/// This is section (B) of `Layer::compatibleSurfaces` filtered back down to
/// what a transport cares about. The filter is `isSensitive()`, not a position
/// in the returned list: that call also returns the layer's approach surfaces,
/// section (A), and its own representing surface, section (C), and which of the
/// three a given entry came from is not recoverable from the order.
///
/// Sections (A) and (C) are no longer discarded. `acceptSurface`
/// (`Layer.cpp:138-151`) lets a non-sensitive surface through only when it
/// carries material, since this call runs with the default
/// `resolveMaterial = true, resolvePassive = false`, so every non-sensitive
/// entry in the returned list is a material crossing of this layer and the
/// second filter below is a guard rather than a selection. They go into
/// `out.material` and change nothing about `out.candidates`, which is what
/// keeps the module answer fixed.
inline LayerResolve onLayer(const Acts::Layer& layer,
                            const Acts::GeometryContext& gctx,
                            const Acts::Vector3& p, const Acts::Vector3& d,
                            double nearLimit,
                            const Acts::BoundaryTolerance& tolerance,
                            const Acts::Surface* skipSurface) {
  Acts::NavigationOptions<Acts::Surface> surfaceOpts;
  surfaceOpts.nearLimit = nearLimit;
  surfaceOpts.startObject = skipSurface;
  surfaceOpts.boundaryTolerance = tolerance;

  const auto surfaces = layer.compatibleSurfaces(gctx, p, d, surfaceOpts);

  LayerResolve out;
  for (const auto& t : surfaces) {
    if (!t.surface().isSensitive()) {
      if (t.surface().surfaceMaterial() != nullptr &&
          t.pathLength() > nearLimit) {
        out.material.push_back(
            {&t.surface(), t.pathLength(), kindOf(t.surface())});
      }
      continue;
    }
    if (t.pathLength() <= nearLimit) {
      continue;
    }
    out.candidates.push_back({&t.surface(), t.pathLength()});
  }
  if (out.candidates.empty()) {
    return out;
  }
  // The tie-break is not cosmetic. `SurfaceArray::populateNeighborCache` sorts
  // each bin's surface pack with `std::ranges::sort` on the pointers, so the
  // order `compatibleSurfaces` hands them back in depends on where the heap put
  // them and changes between two runs of the same binary. Sorting on path alone
  // with a non-stable sort then resolves an exact tie differently from run to
  // run, and the walk names a different module. Measured: with the edge test
  // relaxed to `Infinite()` two runs disagreed on 2 of 50,519 transitions.
  // `geometryId()` is a total order that does not depend on the address space.
  std::sort(out.candidates.begin(), out.candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.path != b.path) {
                return a.path < b.path;
              }
              return a.surface->geometryId().value() <
                     b.surface->geometryId().value();
            });
  out.ok = true;
  out.nearest = out.candidates.front().surface;
  out.nearestPath = out.candidates.front().path;
  return out;
}

/// Ask one volume for the next module, from `p`, along `d`.
///
/// EVERY layer ahead is tried, in path order, not only the nearest. A ray can
/// pass between two modules of the nearest layer, and stopping there sends the
/// caller across a volume boundary and past every remaining layer of this
/// volume. The answer then comes from a volume further out than the one holding
/// the module, which is a wrong answer rather than a missing one, so it would
/// not show up in a count of failures.
inline FirstVolume resolveIn(const Acts::TrackingVolume& volume,
                             const Acts::GeometryContext& gctx,
                             const Acts::Vector3& p, const Acts::Vector3& d,
                             const Acts::Layer* skipLayer,
                             const Acts::Surface* skipSurface,
                             const Options& opts, Result& out) {
  Acts::NavigationOptions<Acts::Layer> layerOpts;
  layerOpts.nearLimit = opts.nearLimit;
  layerOpts.startObject = skipLayer;

  const auto layers = volume.compatibleLayers(gctx, p, d, layerOpts);

  // `isLayerTarget()` is checked rather than assumed. `NavigationTarget::
  // layer()` is `std::get<const Layer*>` on a variant, so calling it on a
  // target that holds something else throws `std::bad_variant_access` out of a
  // `constexpr` accessor, which reads as an ACTS bug rather than as a wrong
  // query here.
  std::vector<const Acts::NavigationTarget*> ordered;
  for (const auto& t : layers) {
    if (!t.isLayerTarget() || t.pathLength() <= opts.nearLimit) {
      continue;
    }
    if (&t.layer() == skipLayer) {
      continue;
    }
    ordered.push_back(&t);
  }
  if (ordered.empty()) {
    return FirstVolume::NoLayer;
  }
  // `compatibleLayers` returns them ordered by path length in the navigator's
  // use. This does not rely on that, because the order is what decides which
  // module is named rather than only which one is reported first.
  std::sort(ordered.begin(), ordered.end(),
            [](const Acts::NavigationTarget* a, const Acts::NavigationTarget* b) {
              return a->pathLength() < b->pathLength();
            });

  // Ask each layer for its modules FROM HERE, standing on the previous module
  // rather than on the layer's approach surface, and take the first layer that
  // has one.
  out.layersAhead.clear();
  for (const Acts::NavigationTarget* candidate : ordered) {
    out.layersAhead.push_back(&candidate->layer());
  }

  for (const Acts::NavigationTarget* candidate : ordered) {
    // THE LAYER'S OWN SURFACE, WHICH `compatibleSurfaces` CANNOT PRODUCE.
    //
    // `Layer::compatibleSurfaces` fast-exits on a layer with no surface array
    // or no approach descriptor (`Layer.cpp:122-125`) and returns nothing at
    // all, section (C) included. Every navigation layer in the ODD is such a
    // layer, and one of them is the PST barrel, which every barrel track
    // crosses on its way from the pixels to the strips. Its material is on the
    // leg and no call this walk made could see it.
    //
    // The stock navigator reaches it the other way round, through
    // `TrackingVolume::compatibleLayers` -> `Layer::surfaceOnApproach`
    // (`Layer.cpp:243-262`), which returns the approach surface when there is
    // an approach descriptor and the representing surface when there is not.
    // That target is this `candidate`, so its surface is exactly what the
    // navigator stops at and what the filter applies material on.
    //
    // A layer that has both a surface array and an approach descriptor offers
    // the same surface again out of section (A). `finalizeMaterial` dedupes by
    // pointer, so the duplicate costs nothing and the slab is applied once.
    if (candidate->surface().surfaceMaterial() != nullptr &&
        candidate->pathLength() > opts.nearLimit) {
      out.material.push_back({&candidate->surface(), candidate->pathLength(),
                              kindOf(candidate->surface())});
    }

    LayerResolve found = onLayer(candidate->layer(), gctx, p, d, opts.nearLimit,
                                 opts.surfaceTolerance, skipSurface);
    // Kept whether or not this layer offered a module. A layer that offers
    // none is still flown through, and there are 89 material representing
    // surfaces against 10 approach ones, so the layers that
    // resolve nothing carry most of the material.
    out.material.insert(out.material.end(), found.material.begin(),
                        found.material.end());
    if (!found.ok) {
      continue;
    }

    out.layer = &candidate->layer();
    out.layerPath = candidate->pathLength();
    // A layer target's `surface()` is what `surfaceOnApproach` intersected,
    // which is the approach surface. NavigationTarget.hpp: the layer
    // constructor takes the surface representation separately from the layer.
    out.layerSurface = &candidate->surface();
    out.queryPos = p;
    out.nearest = found.nearest;
    out.nearestPath = found.nearestPath;
    out.candidates = std::move(found.candidates);
    return FirstVolume::Resolved;
  }

  return FirstVolume::NoModule;
}

/// Put the collected crossings in the shape a caller can fly.
///
/// Three things, and each of them is a way the raw list is wrong:
///
///   A surface at or past the named module is not on this leg. The resolving
///   layer's own approach and representing surfaces come back from
///   `compatibleSurfaces` whether they sit before or after its modules, and the
///   outer one is on the NEXT leg. Applying it here would double count it,
///   because the next walk starts from this module and will offer it again.
///
///   The same surface can arrive twice, once from the layer that owns it and
///   once from a neighbouring layer whose approach descriptor shares it. Two
///   applications of one slab is the failure `determineMaterialUpdateMode`
///   exists to prevent inside ACTS (`PointwiseMaterialInteraction.cpp:16-29`)
///   and the seam is outside it.
///
///   The list arrives in layer order, not in path order, because it is
///   accumulated layer by layer and volume by volume. A caller applying
///   material as it flies needs it sorted.
inline void finalizeMaterial(Result& out, double nearLimit) {
  std::vector<Crossing>& m = out.material;
  m.erase(std::remove_if(m.begin(), m.end(),
                         [&](const Crossing& c) {
                           return c.surface == nullptr ||
                                  c.path <= nearLimit ||
                                  c.path >= out.nearestPath;
                         }),
          m.end());
  std::sort(m.begin(), m.end(), [](const Crossing& a, const Crossing& b) {
    if (a.path != b.path) {
      return a.path < b.path;
    }
    // The same tie-break as the candidate sort and for the same reason: the
    // order two surfaces come back in is not stable between runs.
    return a.surface->geometryId().value() < b.surface->geometryId().value();
  });
  m.erase(std::unique(m.begin(), m.end(),
                      [](const Crossing& a, const Crossing& b) {
                        return a.surface == b.surface;
                      }),
          m.end());
}

}  // namespace detail

/// The walk. Starts in the volume holding `p` and crosses volume boundaries
/// until a module is named, the ray leaves the tracker, or `maxCrossings` runs
/// out.
///
/// Two things move with the walk.
///
/// The query position. Gen1 `compatibleLayers` walks a volume's own
/// `LayerArray` from a starting layer, and picks that start with
/// `associatedLayer(gctx, position)` when no `startObject` is given. A position
/// outside the volume has no meaningful entry in that array, so the query is
/// made from the crossing point and the path lengths are carried back to `p`
/// through an offset. The ray is straight, so that addition is exact rather
/// than an approximation.
///
/// `startObject`. It names where the layer walk begins, not only a surface to
/// skip. The start module's layer belongs to the volume the track is standing
/// in, so handing it to another volume starts that volume's walk outside its
/// own array and the search returns nothing. Both hints are dropped after the
/// first crossing, and the pointer comparison in `resolveIn` still keeps the
/// start layer out of the answer.
///
/// The start layer is excluded, and that is deliberate rather than incidental:
/// it is the whole difference between measuring what a propagator seam could
/// reach and measuring what the stepper seam already gets. Standing on a module
/// of layer L, L's own outer approach surface is a few millimetres ahead and
/// `compatibleLayers` returns it, so keeping L resolves L's own modules again.
/// A caller whose true next surface can be on L has to say so, because this
/// walk will not name it.
inline Result walk(const Acts::TrackingGeometry& tGeo,
                   const Acts::GeometryContext& gctx, const Acts::Vector3& p,
                   const Acts::Vector3& d, const Acts::Layer* startLayer,
                   const Acts::Surface* startSurface, const Options& opts,
                   const Acts::Logger& logger) {
  Result out;

  const Acts::TrackingVolume* vol = tGeo.lowestTrackingVolume(gctx, p);
  if (vol == nullptr) {
    out.stopped = kStoppedNoStartVolume;
    return out;
  }

  // The layer the track is standing on, for its material only.
  //
  // `resolveIn` cannot reach it: `compatibleLayers` takes `startObject` as the
  // layer to start the array walk FROM and skips it (`TrackingVolume.cpp:464`),
  // and that exclusion is deliberate and load-bearing for the module answer.
  // It is wrong for the material. Standing on a module of
  // layer L, the leg still crosses L's outer approach surface and its
  // representing surface if either carries material, and the stock navigator
  // still stops there: those surfaces were resolved into `navSurfaces` when the
  // navigator entered L and are simply the next entries after the module
  // (`Navigator.cpp:436-450`).
  //
  // The candidates this returns go into `startLayerCandidates` and not into
  // `nearest`. Naming a module on the start layer is not what this walk does
  // and putting one in the answer would move the measured rate.
  if (startLayer != nullptr) {
    LayerResolve onStart =
        detail::onLayer(*startLayer, gctx, p, d, opts.nearLimit,
                        opts.surfaceTolerance, startSurface);
    out.material.insert(out.material.end(), onStart.material.begin(),
                        onStart.material.end());
    if (onStart.ok) {
      out.startLayerNearest = onStart.nearest;
      out.startLayerPath = onStart.nearestPath;
      out.startLayerCandidates = std::move(onStart.candidates);
    }
  }

  out.firstVolume =
      detail::resolveIn(*vol, gctx, p, d, startLayer, startSurface, opts, out);
  if (out.firstVolume == FirstVolume::Resolved) {
    out.ok = true;
    detail::finalizeMaterial(out, opts.nearLimit);
    return out;
  }

  Acts::Vector3 q = p;
  double offset = 0.0;
  out.stopped = kStoppedExhausted;

  while (out.crossings < opts.maxCrossings) {
    // The boundary target holds a `BoundarySurface`, which is neither a `Layer`
    // nor a `Surface` in the variant, so only `pathLength()` is read off it.
    // Which volume lies beyond comes from stepping past the crossing and asking
    // the geometry rather than from `attachedVolume`, so this does not depend
    // on the sign convention of the boundary's own orientation.
    Acts::NavigationOptions<Acts::Surface> boundaryOpts;
    boundaryOpts.nearLimit = opts.nearLimit;

    const auto boundaries =
        vol->compatibleBoundaries(gctx, q, d, boundaryOpts, logger);

    const Acts::NavigationTarget* exit = nullptr;
    for (const auto& t : boundaries) {
      if (t.pathLength() <= opts.nearLimit) {
        continue;
      }
      if (exit == nullptr || t.pathLength() < exit->pathLength()) {
        exit = &t;
      }
    }
    if (exit == nullptr) {
      out.stopped = kStoppedNoBoundary;
      return out;
    }

    const Acts::Vector3 beyond = q + (exit->pathLength() + opts.stepPast) * d;
    const Acts::TrackingVolume* next = tGeo.lowestTrackingVolume(gctx, beyond);
    if (next == nullptr || next == vol) {
      out.stopped = kStoppedNoVolume;
      return out;
    }
    // Out of the tracker. Nothing further out is a measurement, so this is the
    // track ending rather than the query failing.
    if (beyond.head<2>().norm() > opts.rMax || std::abs(beyond.z()) > opts.zMax) {
      out.stopped = kStoppedLeftTracker;
      return out;
    }

    // The boundary surface itself, before `offset` moves past it. 24 of the
    // ODD's 123 material surfaces are volume boundaries and this is the only
    // place the walk sees one; `Result::crossed` records the pair of volumes
    // and not the surface between them, which the volume pair alone cannot
    // recover.
    if (exit->surface().surfaceMaterial() != nullptr) {
      out.material.push_back({&exit->surface(), offset + exit->pathLength(),
                              kindOf(exit->surface())});
    }

    out.crossed.emplace_back(vol, next);
    offset += exit->pathLength() + opts.stepPast;
    q = beyond;
    vol = next;
    ++out.crossings;

    // Queried from the crossing point, with both hints dropped, and the paths
    // carried back to `p`.
    const std::size_t matBefore = out.material.size();
    const FirstVolume r =
        detail::resolveIn(*vol, gctx, q, d, nullptr, nullptr, opts, out);
    // Whatever that call appended is measured from `q` and everything already
    // in the list is measured from `p`. Carried back one call at a time, so a
    // walk crossing four boundaries does not accumulate four different origins.
    for (std::size_t i = matBefore; i < out.material.size(); ++i) {
      out.material[i].path += offset;
    }
    if (r == FirstVolume::Resolved) {
      out.layerPath += offset;
      out.nearestPath += offset;
      for (Candidate& c : out.candidates) {
        c.path += offset;
      }
      out.crossingPath = offset;
      out.ok = true;
      detail::finalizeMaterial(out, opts.nearLimit);
      return out;
    }
  }

  return out;
}

/// Stage two of a two-stage resolve: ask one already-chosen layer for its
/// modules, from wherever the caller now stands.
///
/// The stock navigator asks this question from the layer's
/// approach surface, a few millimetres out, and the walk asks it from the
/// previous module, 225 mm out. Both `SurfaceArray::neighbors` and the
/// per-module boundary check degrade with that lever arm. This is the same call
/// with the lever arm removed; what removes it is the caller's business, and
/// nothing here reads a field.
///
/// `nearLimit` is exposed because the caller's position is no longer behind the
/// layer by construction. A curve advanced to the approach surface can land a
/// little past it, and a strictly positive floor would then drop a module the
/// curve has not actually crossed yet.
inline LayerResolve resolveOnLayer(const Acts::Layer& layer,
                                   const Acts::GeometryContext& gctx,
                                   const Acts::Vector3& p,
                                   const Acts::Vector3& d, double nearLimit,
                                   const Acts::BoundaryTolerance& tolerance =
                                       Acts::BoundaryTolerance::None()) {
  return detail::onLayer(layer, gctx, p, d, nearLimit, tolerance, nullptr);
}

}  // namespace nav_walk
