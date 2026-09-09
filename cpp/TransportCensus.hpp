#pragma once

/// The pT distribution of transport calls, split by whether they arrive.
/// HANDOFF items 2 and 2c.
///
/// The latency curve says what one surface-to-surface transport costs. It does
/// not say how many happen or at what momentum, so it cannot be turned into a
/// saving. This counts the second factor.
///
/// What is counted, and it is not the same as the number of steps.
///
/// One transport is one sensitive destination the navigator aims the stepper
/// at. `LearnedStepper::step` consumes a latched target on every call, and with
/// the learned jump off that is several RKN steps per destination, so counting
/// latches would count steps. The census therefore records a destination the
/// first time it is seen on a given propagation and not again, which makes the
/// count one per surface-to-surface transport whichever stepper is inside.
///
/// The steps are counted too, in their own column, because the two questions
/// are different: `n_transport` prices the curve and `n_step` prices whatever
/// the stepper pays per call. A step is counted only when it is aimed at the
/// destination that is being censused, so navigation steps toward portals and
/// layer approaches are not in it.
///
/// Portal and layer-approach surfaces are excluded, because the benchmark's
/// transport is module plane to module plane and the two counts have to be of
/// the same thing to multiply.
///
/// A combinatorial track finder branches, and an abandoned branch has already
/// paid for its transports. Every propagation is counted, surviving or not,
/// which is the point: the cost is paid at the call and not at the track.
///
/// ARRIVAL. HANDOFF item 2c. `Propagator.ipp` fixes what arrival means and
/// makes it observable without a flush at the end of a propagation:
///
///   `getNextTarget` (Propagator.ipp:43-71) calls `updateSurfaceStatus` before
///   any step and returns the target only when the status is `reachable`. A
///   destination the stepper is already on is skipped there and never latched,
///   so every censused transport starts life reachable and no arrival can be
///   recorded before its own transport is.
///
///   After each step the same target is polled again (Propagator.ipp:122). A
///   status of `onSurface` calls `handleSurfaceReached`, and any status other
///   than `reachable` drops the target. So `onSurface` on the destination
///   currently being censused is arrival, and is the only arrival.
///
/// A destination that is dropped, invalidated by `checkTargetValid`, or left
/// pending when the propagation aborts or hits the step limit is simply never
/// marked. `n_transport - n_reached` is therefore the count of transports the
/// finder paid for and did not complete, with no bookkeeping owed at the end of
/// a propagation and nothing to lose if one ends abruptly.
///
/// What is not separable here. Whether a transport belonged to a branch that
/// survived into a final track is not visible from inside the stepper: the
/// stepper is called long before the ambiguity solver decides. The total is
/// what item 1 needs, and the split is reported as not separable.
///
/// Off unless `TRANSPORT_CENSUS` names an output path, so a normal run pays one
/// predictable branch and writes nothing.
///
///     TRANSPORT_CENSUS=/output/runs/x/transport_pt.csv
///
/// The file is written when the process exits.
///
/// How far, and in what field. The two columns below the counts.
///
/// `LearnedTransport.hpp:72` is `kBHelix = 2.0` and both call sites use it, so
/// the helix core runs at a fixed 2 T. The map is not 2 T: over the tracker
/// volume it runs 0.494 to 2.032 T. Whether that matters depends on two
/// numbers nobody had, and both are properties of the steps the navigator
/// actually asks for rather than of the volume:
///
///   PATH. How many millimetres the stepper flies per destination. A wrong
///   field over 3 mm is not the same claim as a wrong field over 300. The
///   accumulator is the step's own return value, which is the 3D path length
///   in both arms (`writeBack` returns `jj.path3d`, `EigenStepper.ipp:368`
///   accumulates the same h), so the two arms are compared on the same
///   quantity and not on a chord against an arc.
///
///   FIELD. The map value at the SOURCE of each destination-aimed step, read
///   from the stepper's own field cache. This is the weight that counts: a
///   grid node is spread evenly through the volume and a step is not, and a
///   module centre is only the second-best proxy for where a step starts.
///
/// Both are histogrammed rather than summed, because the question is a median
/// and a p90 and not a mean. Path is 200 log bins from 0.1 to 10,000 mm, field
/// is 250 linear bins from 0 to 2.5 T, and the exact count and sum are carried
/// beside them so the mean is not a histogram artefact.
///
/// Split by pT slot and, separately, by \|eta\| on the performance writer's own
/// edges (`sim/slice_perf.py`), so the slices line up with efficiency. The
/// field is fine in the barrel and bad forward, and an average over both hides
/// the whole question.
///
/// COVERAGE. A destination's path is credited when the destination is retired,
/// which is either its arrival or the moment a different sensitive destination
/// replaces it. A destination still pending when its propagation ends is never
/// retired and contributes no path record, exactly as it contributes no
/// arrival. `n_path` against `n_transport` in the dump is that coverage and is
/// reported rather than assumed complete.

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace collider_ml {

class TransportCensus {
 public:
  /// HANDOFF item 1's eight bins, so the two measurements multiply directly.
  /// Underflow and overflow are kept rather than clipped: a transport below
  /// 0.5 GeV or above 100 was still paid for, and dropping it would make the
  /// total disagree with the sum of the bins.
  static constexpr int kNBins = 8;
  static constexpr double kEdges[kNBins + 1] = {0.5, 0.7, 1.0,  2.0,
                                                4.0, 8.0, 20.0, 50.0, 100.0};

  /// Slots are the eight bins plus underflow and overflow, so that a caller
  /// holding a slot can be credited later without re-binning a pT that has
  /// meanwhile changed by a few MeV of energy loss.
  static constexpr int kUnder = kNBins;
  static constexpr int kOver = kNBins + 1;
  static constexpr int kNSlots = kNBins + 2;
  static constexpr int kNone = -1;

  /// The performance writer's own \|eta\| edges, so a row here can be put
  /// beside a row of efficiency without rebinning either. `sim/slice_perf.py`
  /// holds the same list. The last slot is everything above 3.0, which the
  /// truth selection cuts but the navigator still steps through.
  static constexpr int kNEtaEdges = 5;
  static constexpr double kEtaEdges[kNEtaEdges] = {0.6, 1.2, 1.8, 2.4, 3.0};
  static constexpr int kNEta = kNEtaEdges + 1;

  /// Path histogram: 200 log bins from 0.1 to 10,000 mm. Five decades at
  /// 10^(1/40) per bin is 5.9 % on a percentile, which separates a 3 mm leg
  /// from a 300 mm one by a wide margin and is the only thing it has to do.
  static constexpr int kNPath = 200;
  static constexpr double kPathLo = -1.0;   ///< log10 mm
  static constexpr double kPathHi = 4.0;

  /// Field histogram: 250 linear bins from 0 to 2.5 T, so 0.01 T per bin and
  /// 0.5 % of the 2.0 T the helix assumes.
  static constexpr int kNField = 250;
  static constexpr double kFieldHi = 2.5;

  static int etaBin(double absEta) {
    for (int i = 0; i < kNEtaEdges; ++i) {
      if (absEta < kEtaEdges[i]) {
        return i;
      }
    }
    return kNEtaEdges;
  }

  static TransportCensus& instance() {
    static TransportCensus c;
    return c;
  }

  bool on() const { return m_on; }

  /// Count one transport. Returns the slot, to be handed back on arrival.
  int record(double pt) {
    m_total.fetch_add(1, std::memory_order_relaxed);
    int slot = kOver;
    if (pt < kEdges[0]) {
      slot = kUnder;
    } else if (pt < kEdges[kNBins]) {
      slot = kNBins - 1;
      for (int i = 0; i < kNBins; ++i) {
        if (pt < kEdges[i + 1]) {
          slot = i;
          break;
        }
      }
    }
    m_n[slot].fetch_add(1, std::memory_order_relaxed);
    return slot;
  }

  /// One step aimed at the transport counted into `slot`.
  void recordStep(int slot) {
    if (slot < 0) {
      return;
    }
    m_step[slot].fetch_add(1, std::memory_order_relaxed);
    m_stepTotal.fetch_add(1, std::memory_order_relaxed);
  }

  /// One step aimed at something that is not a sensitive destination: a portal,
  /// a layer approach, or nothing at all.
  ///
  /// Not a transport and not part of any bin, but it has to be counted
  /// somewhere. The learned jump fires only on sensitive destinations, so these
  /// steps are paid identically by every arm, and the prediction that the
  /// learned arm saves `n_step x (cost of a step)` rests on that. It is an
  /// assumption until this column is compared across two arms, which is what it
  /// is here for.
  void recordOtherStep() {
    m_stepOther.fetch_add(1, std::memory_order_relaxed);
  }

  /// Millimetres flown on a step aimed at something that is not a sensitive
  /// destination. This is the part ACTS keeps in every arm: the learned jump
  /// never fires here, so it is the floor under any saving.
  void recordOtherPath(double mm) {
    addDouble(&m_pathOther, mm);
  }

  /// The map's Bz at the SOURCE of one destination-aimed step, in tesla.
  ///
  /// Per step and not per destination, because it is the field the transport
  /// integrates through and a destination reached in three steps samples the
  /// map three times.
  void recordField(int slot, int eta, double bz) {
    if (slot < 0) {
      return;
    }
    const double a = std::fabs(bz);
    int k = static_cast<int>(a / kFieldHi * kNField);
    if (k < 0) {
      k = 0;
    } else if (k >= kNField) {
      k = kNField - 1;
    }
    m_fieldHist[slot][k].fetch_add(1, std::memory_order_relaxed);
    if (eta >= 0 && eta < kNEta) {
      m_fieldHistEta[eta][k].fetch_add(1, std::memory_order_relaxed);
    }
    m_fieldN.fetch_add(1, std::memory_order_relaxed);
    addDouble(&m_fieldSum, a);
  }

  /// One retired destination and the millimetres spent reaching or failing to
  /// reach it. Called at arrival, and when a different sensitive destination
  /// replaces this one.
  void recordPath(int slot, int eta, double mm) {
    if (slot < 0) {
      return;
    }
    int k = kNPath - 1;
    if (mm > 0.0) {
      const double f = (std::log10(mm) - kPathLo) / (kPathHi - kPathLo);
      k = static_cast<int>(f * kNPath);
      if (k < 0) {
        k = 0;
      } else if (k >= kNPath) {
        k = kNPath - 1;
      }
    } else {
      k = 0;
    }
    m_pathHist[slot][k].fetch_add(1, std::memory_order_relaxed);
    if (eta >= 0 && eta < kNEta) {
      m_pathHistEta[eta][k].fetch_add(1, std::memory_order_relaxed);
    }
    m_pathN[slot].fetch_add(1, std::memory_order_relaxed);
    m_pathNTotal.fetch_add(1, std::memory_order_relaxed);
    addDouble(&m_pathSum, mm);
  }

  /// The transport counted into `slot` reached its destination, after `steps`
  /// steps aimed at it.
  void recordReached(int slot, std::uint64_t steps) {
    if (slot < 0) {
      return;
    }
    m_reached[slot].fetch_add(1, std::memory_order_relaxed);
    m_stepReached[slot].fetch_add(steps, std::memory_order_relaxed);
    m_reachedTotal.fetch_add(1, std::memory_order_relaxed);
    m_stepReachedTotal.fetch_add(steps, std::memory_order_relaxed);
  }

  ~TransportCensus() { dump(); }

  void dump() const {
    if (!m_on || m_path.empty()) {
      return;
    }
    std::FILE* f = std::fopen(m_path.c_str(), "w");
    if (f == nullptr) {
      return;
    }
    std::fprintf(f, "pt_lo,pt_hi,n_transport,n_reached,n_step,n_step_reached\n");
    // Underflow and overflow carry the edge they fell outside, so a reader can
    // see which side they are on without a separate convention.
    row(f, "", kEdges[0], kUnder);
    for (int i = 0; i < kNBins; ++i) {
      std::fprintf(f, "%g,%g,", kEdges[i], kEdges[i + 1]);
      counts(f, i);
    }
    std::fprintf(f, "%g,,", kEdges[kNBins]);
    counts(f, kOver);
    // Steps toward a non-sensitive target. No pT bin, no transport and no
    // arrival, so only the step column is filled and the other three are
    // blank rather than zero.
    std::fprintf(f, "other,,,,%llu,\n", u(m_stepOther));
    std::fprintf(f, "total,,%llu,%llu,%llu,%llu\n", u(m_total), u(m_reachedTotal),
                 u(m_stepTotal), u(m_stepReachedTotal));
    std::fclose(f);
    std::fprintf(stderr,
                 "[census] %llu transport calls, %llu reached, %llu steps -> %s\n",
                 u(m_total), u(m_reachedTotal), u(m_stepTotal), m_path.c_str());
    dumpDist();
  }

 private:
  using Counter = std::atomic<std::uint64_t>;

  static unsigned long long u(const Counter& c) {
    return static_cast<unsigned long long>(c.load(std::memory_order_relaxed));
  }

  /// A compare-exchange add on a double.
  ///
  /// `std::atomic<double>::fetch_add` is C++20 and the image's libstdc++ has
  /// it, but the loop is three lines and does not depend on that, and these
  /// are sums of millimetres taken once per step where the surrounding work is
  /// a Runge-Kutta stage. It costs nothing that is measurable here.
  static void addDouble(std::atomic<double>* dst, double v) {
    double cur = dst->load(std::memory_order_relaxed);
    while (!dst->compare_exchange_weak(cur, cur + v,
                                       std::memory_order_relaxed)) {
    }
  }

  /// Percentile-bearing rows for one histogram, in long format.
  ///
  /// One row per non-empty bin. Reconstructing a percentile from bin edges is
  /// the reader's job (`src/prop/census_paths.py`), which keeps the format
  /// free of any choice about which percentiles matter.
  template <std::size_t N>
  void hist(std::FILE* f, const char* metric, const char* split,
            const char* slice, const std::array<Counter, N>& h, double lo,
            double hi, bool log) const {
    for (std::size_t k = 0; k < N; ++k) {
      const unsigned long long n = u(h[k]);
      if (n == 0) {
        continue;
      }
      const double a = lo + (hi - lo) * static_cast<double>(k) / N;
      const double b = lo + (hi - lo) * static_cast<double>(k + 1) / N;
      std::fprintf(f, "%s,%s,%s,%.6g,%.6g,%llu\n", metric, split, slice,
                   log ? std::pow(10.0, a) : a, log ? std::pow(10.0, b) : b, n);
    }
  }

  void dumpDist() const {
    // <stem>_dist.csv beside the counts, rather than more columns on a file
    // whose rows are pT bins: these rows are histogram bins and the two do not
    // share a key.
    std::string p = m_path;
    const std::size_t dot = p.rfind(".csv");
    p = (dot == std::string::npos ? p : p.substr(0, dot)) + "_dist.csv";
    std::FILE* f = std::fopen(p.c_str(), "w");
    if (f == nullptr) {
      return;
    }
    std::fprintf(f, "metric,split,slice,lo,hi,count\n");
    char s[64];
    for (int i = 0; i < kNSlots; ++i) {
      slotName(s, sizeof(s), i);
      hist(f, "path", "pt", s, m_pathHist[i], kPathLo, kPathHi, true);
      hist(f, "field", "pt", s, m_fieldHist[i], 0.0, kFieldHi, false);
    }
    for (int i = 0; i < kNEta; ++i) {
      etaName(s, sizeof(s), i);
      hist(f, "path", "eta", s, m_pathHistEta[i], kPathLo, kPathHi, true);
      hist(f, "field", "eta", s, m_fieldHistEta[i], 0.0, kFieldHi, false);
    }
    // Exact scalars, so a mean is never read off the histogram.
    std::fprintf(f, "scalar,all,n_transport,,,%llu\n", u(m_total));
    std::fprintf(f, "scalar,all,n_path,,,%llu\n", u(m_pathNTotal));
    std::fprintf(f, "scalar,all,n_field,,,%llu\n", u(m_fieldN));
    std::fprintf(f, "scalar,all,path_sum_mm,,,%.6f\n",
                 m_pathSum.load(std::memory_order_relaxed));
    std::fprintf(f, "scalar,all,path_other_sum_mm,,,%.6f\n",
                 m_pathOther.load(std::memory_order_relaxed));
    std::fprintf(f, "scalar,all,field_sum_T,,,%.6f\n",
                 m_fieldSum.load(std::memory_order_relaxed));
    std::fclose(f);
    std::fprintf(stderr,
                 "[census] %llu path records over %llu transports, "
                 "%llu field samples -> %s\n",
                 u(m_pathNTotal), u(m_total), u(m_fieldN), p.c_str());
  }

  static void slotName(char* s, std::size_t n, int i) {
    if (i == kUnder) {
      std::snprintf(s, n, "under_%g", kEdges[0]);
    } else if (i == kOver) {
      std::snprintf(s, n, "over_%g", kEdges[kNBins]);
    } else {
      std::snprintf(s, n, "%g-%g", kEdges[i], kEdges[i + 1]);
    }
  }

  static void etaName(char* s, std::size_t n, int i) {
    if (i == 0) {
      std::snprintf(s, n, "0-%g", kEtaEdges[0]);
    } else if (i == kNEtaEdges) {
      std::snprintf(s, n, "over_%g", kEtaEdges[kNEtaEdges - 1]);
    } else {
      std::snprintf(s, n, "%g-%g", kEtaEdges[i - 1], kEtaEdges[i]);
    }
  }

  void counts(std::FILE* f, int slot) const {
    std::fprintf(f, "%llu,%llu,%llu,%llu\n", u(m_n[slot]), u(m_reached[slot]),
                 u(m_step[slot]), u(m_stepReached[slot]));
  }

  void row(std::FILE* f, const char* lo, double hi, int slot) const {
    std::fprintf(f, "%s,%g,", lo, hi);
    counts(f, slot);
  }

  TransportCensus() {
    const char* p = std::getenv("TRANSPORT_CENSUS");
    if (p != nullptr && *p != '\0') {
      m_path = p;
      m_on = true;
    }
  }

  bool m_on = false;
  std::string m_path;
  std::array<Counter, kNSlots> m_n{};
  std::array<Counter, kNSlots> m_reached{};
  std::array<Counter, kNSlots> m_step{};
  std::array<Counter, kNSlots> m_stepReached{};
  Counter m_total{0};
  Counter m_reachedTotal{0};
  Counter m_stepTotal{0};
  Counter m_stepReachedTotal{0};
  Counter m_stepOther{0};

  std::array<std::array<Counter, kNPath>, kNSlots> m_pathHist{};
  std::array<std::array<Counter, kNPath>, kNEta> m_pathHistEta{};
  std::array<std::array<Counter, kNField>, kNSlots> m_fieldHist{};
  std::array<std::array<Counter, kNField>, kNEta> m_fieldHistEta{};
  std::array<Counter, kNSlots> m_pathN{};
  Counter m_pathNTotal{0};
  Counter m_fieldN{0};
  std::atomic<double> m_pathSum{0.0};
  std::atomic<double> m_pathOther{0.0};
  std::atomic<double> m_fieldSum{0.0};
};

}  // namespace collider_ml
