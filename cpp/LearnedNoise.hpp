// Q, the process noise of the learned transport, in the form the covariance
// gets it added in.
//
// F is computed and Q is measured. `LearnedJacobian.hpp` is the first half;
// this is the second. Q is what the model gets wrong over one jump, measured on
// held-out teacher jumps whose incoming uncertainty is exactly zero because
// every training jump starts from the true state. It cannot be
// derived and it is not the stepper's to invent, so this header holds a lookup
// and no physics.
//
// Five things that are easy to get wrong here, in order of how quiet they are:
//
//  1. Q must enter once per jump. `transportCovarianceToBound` is not called
//     once per step; the CKF calls it at CombinatorialKalmanFilter.hpp:419,
//     :481 and :612, and a propagation can also reach it through boundState.
//     So the jump arms a flag and the first transport consumes it, the same
//     shape as the target latch in LearnedStepper::step. Adding Q twice
//     inflates the covariance by exactly the amount that makes the filter look
//     stable, which is the worst possible direction for a bug to fail in.
//
//  2. The MATERIAL-OFF table. The material-on measurement contains multiple
//     scattering, and ACTS adds scattering again at every material surface
//     (PointwiseMaterialInteraction.hpp:213-221). Using the material-on table
//     double counts it. The split that keeps the two separable is that this Q
//     is what the propagation model gets wrong and ACTS's is scattering.
//
//  3. The shape. This Q is 5x5 over (loc0, loc1, phi, theta, q/p) with position
//     variance and a -0.86 position-direction correlation. ACTS's own noise
//     model has three diagonal entries and neither of those things. It is not
//     reshaped to fit; it goes into the top-left 5x5 of the BoundMatrix and the
//     time row and column are left alone, because nothing here measured them.
//     The correlation is load-bearing: silicon measures position and never
//     direction, so the correlation is the only route by which measuring a
//     position can fix a wrong direction, and a diagonal Q leaves a chained
//     filter with no brake.
//
//  4. UNITS. The table is written in the units the residuals were measured in,
//     micrometres and milliradians, and ACTS is mm and rad. A variance carries
//     the square of the conversion. Getting this wrong by 1e-3 rather than
//     1e-6 gives a Q that is a thousand times too big and a filter that
//     accepts everything.
//
//  5. The sigma head scales sigma, so it enters the VARIANCE squared, and it
//     scales the whole 5x5 rather than the diagonal, or it would change the
//     correlations it was not measured to change.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace collider_ml {

using Mat55 = Eigen::Matrix<double, 5, 5>;

/// The binning, identical to `chi2_gate.PT_EDGES` / `ETA_EDGES`. Written out
/// rather than read from the file so that a table built against a different
/// binning fails the header check instead of being silently mis-indexed.
inline constexpr int kNPt = 5;
inline constexpr int kNEta = 5;
inline constexpr double kPtEdges[kNPt + 1] = {0.0, 1.0, 2.0, 5.0, 10.0, 1e9};
inline constexpr double kEtaEdges[kNEta + 1] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};

/// Module classes, in the order `qtable.CLASSES` writes them.
enum class ModuleClass : int { kPixel = 0, kShortStrip = 1, kLongStrip = 2 };
inline constexpr int kNClass = 3;

/// The ODD volume ids, from `chi2_gate.CLASS_OF`. A volume this does not know
/// is not guessed at: `classOf` reports failure and the caller adds no noise
/// rather than the wrong noise.
inline bool classOf(unsigned volume, ModuleClass* out) {
  switch (volume) {
    case 16: case 17: case 18:
      *out = ModuleClass::kPixel;
      return true;
    case 23: case 24: case 25:
      *out = ModuleClass::kShortStrip;
      return true;
    case 28: case 29: case 30:
      *out = ModuleClass::kLongStrip;
      return true;
    default:
      return false;
  }
}

inline int ptBin(double pt) {
  for (int i = kNPt - 1; i > 0; --i) {
    if (pt >= kPtEdges[i]) {
      return i;
    }
  }
  return 0;
}

inline int etaBin(double absEta) {
  for (int i = kNEta - 1; i > 0; --i) {
    if (absEta >= kEtaEdges[i]) {
      return i;
    }
  }
  return 0;
}

/// Which transport the jump was made by. The field gate in
/// `LearnedStepper::step` runs the network above the threshold and the helix
/// alone at or below it, and those are two different models with two different
/// errors. A table measured on one of them says nothing about the other.
enum class Branch : int { kFired = 0, kHelixAlone = 1 };
inline constexpr int kNBranch = 2;

/// The measured process noise, one 5x5 per (class, pt bin, |eta| bin), plus a
/// per-class fallback for cells the measurement did not fill.
///
/// Loaded from a file rather than compiled in, unlike the network weights. The
/// weights are the model and the port means nothing without them; Q is a
/// measurement that is re-taken whenever the target surface or the training set
/// changes, and a header would go stale silently.
///
/// A table carries one branch or two. A one-branch table is the format every
/// table before this one was written in and it arms the same Q on both
/// branches, which measures a pixel loc1 pull of 46 where the helix runs alone
/// against 0.39 where the network fires. It is
/// still loaded, and it still behaves exactly as it did, so an old table and an
/// old measurement remain comparable.
class NoiseTable {
 public:
  /// Reads what `src/prop/export_qtable.py` writes. Returns false and says why
  /// on stderr rather than throwing, because the caller's honest response is to
  /// add no noise and be loud about it.
  ///
  /// `weightsMd5` is the md5 of the `cpp/gtheta_weights.hpp` this binary was
  /// compiled from, which `cpp/incontainer_build_ckf.sh` computes at the build
  /// and passes in as `GTHETA_WEIGHTS_MD5`. `fieldGate` is the threshold this
  /// run is about to arm the stepper's gate at. Both are recorded in the file
  /// by the exporter and both are compared here, for exactly the reason the
  /// material flag is compared: a table measured on another model, or at
  /// another threshold, is a covariance for a runtime path this run does not
  /// take, and nothing about the numbers says so. One such mismatch has been
  /// measured: a table measured with the network firing on
  /// every transport, armed on a branch where it never fires, and a pixel loc1
  /// pull of 46 against a declared 1.
  bool load(const std::string& path, const std::string& weightsMd5,
            double fieldGate) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      std::fprintf(stderr, "NoiseTable: cannot open %s\n", path.c_str());
      return false;
    }
    std::int32_t magic = 0, nClass = 0, nPt = 0, nEta = 0, material = 0;
    bool ok = std::fread(&magic, 4, 1, f) == 1 &&
              std::fread(&nClass, 4, 1, f) == 1 &&
              std::fread(&nPt, 4, 1, f) == 1 &&
              std::fread(&nEta, 4, 1, f) == 1 &&
              std::fread(&material, 4, 1, f) == 1;
    if (!ok ||
        (magic != kMagic && magic != kMagicBranch && magic != kMagicProv)) {
      std::fprintf(stderr, "NoiseTable: %s is not a Q table\n", path.c_str());
      std::fclose(f);
      return false;
    }
    // A table written before the provenance fields existed cannot be checked
    // against anything, and a covariance that cannot be checked is the defect
    // this refusal exists to stop. Refused rather than warned about, the same
    // way material-on is refused: the two tables are indistinguishable by
    // inspection and the wrong one fails loudly nowhere else.
    if (magic != kMagicProv) {
      std::fprintf(stderr,
                   "NoiseTable: %s records no provenance. Re-export it with "
                   "prop.export_qtable --weights <the gtheta_weights.hpp the "
                   "residuals were taken from> --field-gate <the threshold the "
                   "dump was made at>.\n",
                   path.c_str());
      std::fclose(f);
      return false;
    }
    // The branch count is in the file rather than inferred from its length,
    // for the reason the material flag is: the two tables are the same size
    // per branch and a mis-read would be a silently wrong covariance.
    std::int32_t nBranch = 1;
    char fileMd5[32] = {0};
    double fileGate = 0.0;
    if (std::fread(&nBranch, 4, 1, f) != 1 ||
        std::fread(fileMd5, 1, 32, f) != 32 ||
        std::fread(&fileGate, sizeof(double), 1, f) != 1) {
      ok = false;
    }
    if (!ok || nBranch < 1 || nBranch > kNBranch) {
      std::fprintf(stderr, "NoiseTable: %s declares %d branches, this build "
                   "holds 1 to %d\n", path.c_str(), nBranch, kNBranch);
      std::fclose(f);
      return false;
    }
    if (nClass != kNClass || nPt != kNPt || nEta != kNEta) {
      std::fprintf(stderr,
                   "NoiseTable: %s is binned %dx%dx%d, this build expects "
                   "%dx%dx%d\n",
                   path.c_str(), nClass, nPt, nEta, kNClass, kNPt, kNEta);
      std::fclose(f);
      return false;
    }
    // The material flag is in the file because the difference between the two
    // tables is invisible in the numbers and catastrophic in the filter. A
    // material-on table double counts scattering with ACTS's own.
    if (material != 0) {
      std::fprintf(stderr,
                   "NoiseTable: %s was measured with material ON. ACTS adds "
                   "scattering again at every material surface, so this would "
                   "double count it. Use the material-off table.\n",
                   path.c_str());
      std::fclose(f);
      return false;
    }
    // The weights the residuals were taken from. An empty string on this side
    // is a build that did not record its own, which is a build whose
    // covariance cannot be checked, so it is refused too rather than waved
    // through.
    if (weightsMd5.size() != 32 ||
        std::memcmp(fileMd5, weightsMd5.data(), 32) != 0) {
      std::fprintf(stderr,
                   "NoiseTable: %s was measured against weights %.32s and this "
                   "binary holds %s. A table is a measurement of one model's "
                   "error and says nothing about another's.\n",
                   path.c_str(), fileMd5,
                   weightsMd5.empty() ? "no recorded md5" : weightsMd5.c_str());
      std::fclose(f);
      return false;
    }
    // The field gate the dump was made at. Above the threshold the network
    // runs and at or below it the helix runs alone, so the threshold decides
    // which transport made each jump the table was measured on, and a table
    // measured at one threshold describes a different population at another.
    if (std::abs(fileGate - fieldGate) > 1e-9) {
      std::fprintf(stderr,
                   "NoiseTable: %s was measured at FIELD_GATE=%.6g and this run "
                   "arms the gate at %.6g. The threshold decides which "
                   "transport made each jump the table was measured on.\n",
                   path.c_str(), fileGate, fieldGate);
      std::fclose(f);
      return false;
    }
    m_weightsMd5.assign(fileMd5, 32);
    m_fieldGate = fileGate;
    m_nBranch = nBranch;
    const std::size_t nCell =
        static_cast<std::size_t>(nBranch) * kNClass * kNPt * kNEta;
    m_cell.assign(nCell, Mat55::Zero());
    m_have.assign(nCell, 0);
    m_fallback.assign(static_cast<std::size_t>(nBranch) * kNClass,
                      Mat55::Zero());
    std::vector<double> buf(25);
    for (std::size_t c = 0; ok && c < m_fallback.size(); ++c) {
      if (std::fread(buf.data(), sizeof(double), 25, f) != 25) {
        ok = false;
        break;
      }
      m_fallback[c] = Eigen::Map<const Eigen::Matrix<double, 5, 5,
                                                     Eigen::RowMajor>>(
          buf.data());
    }
    for (std::size_t i = 0; ok && i < nCell; ++i) {
      std::uint8_t have = 0;
      if (std::fread(&have, 1, 1, f) != 1 ||
          std::fread(buf.data(), sizeof(double), 25, f) != 25) {
        ok = false;
        break;
      }
      m_have[i] = have;
      m_cell[i] = Eigen::Map<const Eigen::Matrix<double, 5, 5,
                                                 Eigen::RowMajor>>(buf.data());
    }
    std::fclose(f);
    if (!ok) {
      std::fprintf(stderr, "NoiseTable: %s is truncated\n", path.c_str());
      m_cell.clear();
      return false;
    }
    m_loaded = true;
    return true;
  }

  bool loaded() const { return m_loaded; }

  /// How many branches this table was measured on. One means the same Q on
  /// both, which is every table written before the branch axis existed.
  int branches() const { return m_nBranch; }

  /// What the file says it was measured against. Printed by the factory so the
  /// run's own log carries it, which is the record `cpp/gtheta_weights.hpp`'s
  /// `source:` comment was being asked to be and is not.
  const std::string& weightsMd5() const { return m_weightsMd5; }
  double fieldGate() const { return m_fieldGate; }

  /// Q for one jump, in ACTS units, already scaled by the sigma head.
  ///
  /// `m` is `Prediction::m`, the per-hit sigma SCALE. It multiplies sigma, so
  /// it enters the variance squared, and it multiplies the whole matrix so the
  /// measured correlations are preserved rather than quietly changed.
  ///
  /// `branch` is which transport made the jump. A one-branch table ignores it
  /// and returns what it always returned, so an old table on this build is the
  /// same measurement it was on the old one.
  bool lookup(ModuleClass cls, double pt, double absEta, double m,
              Branch branch, Mat55* out) const {
    if (!m_loaded) {
      return false;
    }
    const int c = static_cast<int>(cls);
    const int b = (m_nBranch > 1) ? static_cast<int>(branch) : 0;
    const std::size_t i =
        ((static_cast<std::size_t>(b) * kNClass + c) * kNPt + ptBin(pt)) *
            kNEta +
        etaBin(absEta);
    // A cell the measurement never filled falls back to the class value, which
    // is what `chi2_gate.measured_cov` does with its pt_bin = eta_bin = -1 row.
    // Interpolating between neighbouring cells would be inventing a
    // measurement. The fallback is per branch for the same reason the cell is.
    const std::size_t fb = static_cast<std::size_t>(b) * kNClass + c;
    *out = (m_have[i] != 0 ? m_cell[i] : m_fallback[fb]) * (m * m);
    return true;
  }

 private:
  static constexpr std::int32_t kMagic = 0x51544142;        // "QTAB"
  static constexpr std::int32_t kMagicBranch = 0x32425451;  // "QTB2"
  static constexpr std::int32_t kMagicProv = 0x33425451;    // "QTB3"
  bool m_loaded = false;
  int m_nBranch = 1;
  std::string m_weightsMd5;
  double m_fieldGate = 0.0;
  std::vector<Mat55> m_cell;
  std::vector<std::uint8_t> m_have;
  std::vector<Mat55> m_fallback;
};

}  // namespace collider_ml
