//==========================================================================
// A DD4hep magnetic field that reads the ODD's own xyz field map.
//
// WHY THIS EXISTS. ODD v4.0.4 ships `data/odd-bfield.csv`, a 201 x 201 x 301
// xyz grid, and its compact does not reference it: `OpenDataDetector.xml`
// defines `<field type="solenoid">`, which is analytic and piecewise uniform,
// 2 T inside r = 1160 mm and |z| = 3 m. The tracker sits entirely inside that,
// so `detector.field` reads exactly 2.0000 T with no transverse component
// anywhere a track goes. Both stages of the chain take their field from the
// compact: `ddsim_run.py:146` and `digi_and_reco.py:167` each resolve it
// through `getOpenDataDetectorDirectory()`. Changing the compact is therefore
// the only place a field change reaches simulation and reconstruction at once.
//
// DD4hep 1.32 as built in the pinned image registers four XML field elements
// and none of them reads a grid from a file. Verified by dumping the factory
// symbols out of `libDDCorePlugins.so` and `libDDDetectors.so`:
// `ConstantField`, `solenoid`/`SolenoidMagnet`, `DipoleMagnet`,
// `MultipoleMagnet`. `lcgeo` and `k4geo`, which ship `FieldMapBrBz` and
// `FieldMapXYZ`, are not installed. So the plugin is written rather than
// selected.
//
// The detector this makes is not the ODD. A detector whose field comes from
// the map is not the one the ODD release describes, and no number taken in it
// is comparable to a published ODD baseline.
//
// USAGE, in the compact, in place of the `<field type="solenoid">` element:
//
//   <field type="ODDFieldMapXyz" name="GlobalFieldMap"
//          file="/opt/odd/data/odd-bfield-xyz.bin"/>
//
// The file is the binary `src/prop/export_fieldbin.py --components xyz`
// writes from the csv. The csv itself is not read here: it is 515 MB of ascii
// and parsing it costs more than a Geant4 event, at every job start, in every
// stage.
//==========================================================================
#include <DD4hep/DetFactoryHelper.h>
#include <DD4hep/FieldTypes.h>
#include <DD4hep/Printout.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

/// Trilinear interpolation on the ODD's regular xyz grid.
///
/// UNITS, and this is the one thing here with no test behind it. DD4hep's
/// default unit system is not Geant4's: `Evaluator/DD4hepUnits.h:38-41` sets
/// `millimeter = 0.1` unless `DD4HEP_USE_GEANT4_UNITS` is defined, so the
/// `pos` this is handed is in centimetres-as-1.0, and `tesla` is a derived
/// constant rather than 1. Both conversions below therefore go through the
/// header's own constants rather than through a literal, which is correct
/// under either setting. The grid's own numbers are millimetres and tesla,
/// because that is what the csv holds.
///
/// This is the exact shape of the defect that put the network's field input at
/// -9.99 standard deviations for a month: a quantity in one unit system read
/// as though it were in another, with every value still looking plausible. No
/// test catches it. The gate that catches it is reading the field back at
/// known points, which `sim/odd_map/field_gate.py` does.
class ODDFieldMapXyz : public dd4hep::CartesianField::Object {
 public:
  explicit ODDFieldMapXyz(const std::string& path) { load(path); }

  /// Add this field's contribution at `pos`.
  ///
  /// The base class contract (`DD4hep/Fields.h`) is that components are ADDED,
  /// so that several fields can be overlaid. Hence `+=` and not `=`.
  void fieldComponents(const double* pos, double* field) override {
    // DD4hep length -> mm. `dd4hep::mm` is 0.1 in the default system.
    const double p[3] = {pos[0] / dd4hep::mm, pos[1] / dd4hep::mm,
                         pos[2] / dd4hep::mm};

    double b[3] = {0.0, 0.0, 0.0};
    if (!sample(p, b)) {
      // OUT OF RANGE. Nothing is added, which leaves the caller's accumulator
      // untouched and is the same answer the analytic solenoid gives outside
      // its own envelope.
      //
      // ACTS does something different with the same map: its
      // `InterpolatedBFieldMap::getField` returns
      // `MagneticFieldError::OutOfBounds` (`InterpolatedBFieldMap.hpp:248`), a
      // failed `Result` rather than a value. `fieldComponents` returns void
      // and cannot express that, so the two cannot be made to agree here.
      //
      // It should not be reachable in this detector: the grid spans
      // x, y in [-10, 10] m and z in [-15, 15] m, and `world_size` is 10 m.
      return;
    }

    // tesla -> DD4hep field units, at each of the three sites.
    field[0] += b[0] * dd4hep::tesla;
    field[1] += b[1] * dd4hep::tesla;
    field[2] += b[2] * dd4hep::tesla;
  }

 private:
  /// `false` if the point is outside the grid, in which case `b` is untouched.
  bool sample(const double* p, double* b) const {
    int i0[3];
    double t[3];
    for (int d = 0; d < 3; ++d) {
      const double f = (p[d] - m_o[d]) / m_h[d];
      if (f < 0.0 || f > static_cast<double>(m_n[d] - 1)) {
        return false;
      }
      // The cell index is clamped to n-2 so that a point exactly on the upper
      // grid face interpolates inside the last cell rather than falling out of
      // it. ACTS excludes that face instead (`isInsideLocal` uses `>=`); the
      // difference is one node at the edge of the world and is recorded here
      // rather than left to be discovered.
      int i = static_cast<int>(std::floor(f));
      if (i < 0) {
        i = 0;
      } else if (i > m_n[d] - 2) {
        i = m_n[d] - 2;
      }
      i0[d] = i;
      t[d] = f - static_cast<double>(i);
    }

    for (int dx = 0; dx < 2; ++dx) {
      const double wx = dx ? t[0] : 1.0 - t[0];
      for (int dy = 0; dy < 2; ++dy) {
        const double wy = dy ? t[1] : 1.0 - t[1];
        for (int dz = 0; dz < 2; ++dz) {
          const double wz = dz ? t[2] : 1.0 - t[2];
          const double w = wx * wy * wz;
          // x outer, y middle, z inner, three components interleaved. Same
          // ordering the csv has and `src/prop/build_fieldmap.py` checks
          // rather than assumes.
          const std::size_t k =
              ((static_cast<std::size_t>(i0[0] + dx) * m_n[1] +
                static_cast<std::size_t>(i0[1] + dy)) *
                   m_n[2] +
               static_cast<std::size_t>(i0[2] + dz)) *
              3;
          b[0] += w * m_b[k];
          b[1] += w * m_b[k + 1];
          b[2] += w * m_b[k + 2];
        }
      }
    }
    return true;
  }

  /// A missing or short file throws. It never falls back to a constant.
  ///
  /// The layout is `src/prop/export_fieldbin.py`'s, which `cpp/bench_kernel.cpp`
  /// already reads in its one-component form:
  ///
  ///     int32   n[3]                nx, ny, nz
  ///     double  o[3]                grid origin, mm
  ///     double  h[3]                spacing, mm
  ///     float   b[nx*ny*nz*3]       Bx, By, Bz interleaved, tesla
  ///
  /// There is no magic number in that layout, so the size is the check: a
  /// one-component file of the same grid is exactly a third of the payload and
  /// would otherwise be read as a third of a map.
  void load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      throw std::runtime_error("ODDFieldMapXyz: cannot open " + path);
    }
    std::int32_t n[3];
    double o[3];
    double h[3];
    if (std::fread(n, sizeof(std::int32_t), 3, f) != 3 ||
        std::fread(o, sizeof(double), 3, f) != 3 ||
        std::fread(h, sizeof(double), 3, f) != 3) {
      std::fclose(f);
      throw std::runtime_error("ODDFieldMapXyz: short header in " + path);
    }
    for (int d = 0; d < 3; ++d) {
      if (n[d] < 2) {
        std::fclose(f);
        throw std::runtime_error("ODDFieldMapXyz: degenerate grid in " + path);
      }
      if (!(h[d] > 0.0)) {
        std::fclose(f);
        throw std::runtime_error("ODDFieldMapXyz: bad spacing in " + path);
      }
      m_n[d] = n[d];
      m_o[d] = o[d];
      m_h[d] = h[d];
    }

    const std::size_t nodes = static_cast<std::size_t>(m_n[0]) *
                              static_cast<std::size_t>(m_n[1]) *
                              static_cast<std::size_t>(m_n[2]);
    m_b.resize(nodes * 3);
    const std::size_t got = std::fread(m_b.data(), sizeof(float), nodes * 3, f);
    // One extra byte means the file is not what the header says it is.
    const bool trailing = (std::fgetc(f) != EOF);
    std::fclose(f);
    if (got != nodes * 3 || trailing) {
      throw std::runtime_error(
          "ODDFieldMapXyz: " + path + " is not a 3-component map of " +
          std::to_string(m_n[0]) + "x" + std::to_string(m_n[1]) + "x" +
          std::to_string(m_n[2]) + " (read " + std::to_string(got) +
          " of " + std::to_string(nodes * 3) + " floats)");
    }

    // Printed once per job, because the alternative is a detector whose field
    // nobody can name from the log.
    //
    // WARNING and not INFO, which is not a claim that anything is wrong. The
    // production chain builds the detector through `acts.examples.odd`, and
    // `odd.py` passes `dd4hepLogLevel=customLogLevel(minLevel=WARNING)`, so an
    // INFO printout is filtered out of every ddsim and digi_and_reco job. The
    // one configuration this experiment must never run unnoticed is a
    // map-simulated sample reconstructed in the uniform-field geometry, and
    // the only cheap check against it is a line in the log of the job itself.
    // A message that is invisible exactly where it is needed is not a check.
    dd4hep::printout(dd4hep::WARNING, "ODDFieldMapXyz",
                     "%s: %dx%dx%d, origin %.0f/%.0f/%.0f mm, spacing "
                     "%.0f/%.0f/%.0f mm, %zu nodes",
                     path.c_str(), m_n[0], m_n[1], m_n[2], m_o[0], m_o[1],
                     m_o[2], m_h[0], m_h[1], m_h[2], nodes);
    dd4hep::printout(dd4hep::WARNING, "ODDFieldMapXyz",
                     "units: 1 mm = %g, 1 tesla = %g in this DD4hep build",
                     dd4hep::mm, dd4hep::tesla);
  }

  std::int32_t m_n[3] = {0, 0, 0};
  double m_o[3] = {0.0, 0.0, 0.0};  ///< mm
  double m_h[3] = {0.0, 0.0, 0.0};  ///< mm
  std::vector<float> m_b;           ///< 3 * nodes, tesla
};

/// `<field type="ODDFieldMapXyz" name="..." file="..."/>`
///
/// Signature fixed by `DECLARE_XMLELEMENT` (`DD4hep/Factories.h:301-303`):
/// `Ref_t (*)(Detector&, xml::Handle_t)`.
dd4hep::Ref_t create_ODDFieldMapXyz(dd4hep::Detector& /* description */,
                                    dd4hep::xml::Handle_t handle) {
  dd4hep::xml::Component c(handle);
  const std::string name = c.attr<std::string>(_Unicode(name));
  const std::string file = c.attr<std::string>(_Unicode(file));

  auto* obj = new ODDFieldMapXyz(file);
  obj->field_type = dd4hep::CartesianField::MAGNETIC;

  // `assign` and not a constructor: `CartesianField` is a `Handle<NamedObject>`
  // and takes another handle, never a raw pointer (`DD4hep/Fields.h:86-93`).
  // `Handle::assign` (`DD4hep/Handle.h:173`) is the one that adopts a new
  // object and sets its name and title.
  dd4hep::CartesianField field;
  field.assign(obj, name, "ODDFieldMapXyz");
  return field;
}

}  // namespace

DECLARE_XMLELEMENT(ODDFieldMapXyz, create_ODDFieldMapXyz)
