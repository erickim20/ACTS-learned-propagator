// The per-jump sigma the network's position outputs are written in.
//
// `train_gtheta.py` trains outputs 0 and 1 on `residual / sigma`, where sigma
// is that jump's own, `hypot(module resolution, measured covariance width)`
// (`:600-607`). So an output of 1 means "one sigma of THIS module at THIS pT
// and |eta|", and the runtime has to multiply by the same quantity or the
// correction the model asked for is not the correction the filter receives.
//
// It did not. `learned_ckf.cpp` set two constants once per run, 20 um and
// 43 um, and `LearnedStepper::toJump` wrote them onto every jump. The ODD's own
// digitisation is 15/15 on pixels, 43/1200 on short strips and 72/1D on long
// strips, so the applied scale was 2.9x too large on a pixel's loc1 and 28x too
// small on a short strip's. `src/prop/export_cell_sigma.py` writes what this
// loads and carries the rest of the argument.
//
// Three things that are easy to get wrong here:
//
//  1. The table is in micrometres, because that is what the residual was
//     measured in and what the model was trained against. `LearnedTransport`
//     multiplies by `detail::kMM` at the point of use. This header does not
//     convert, so a reader comparing it against an ACTS length is off by 1e3.
//
//  2. sig1 = 0 on long strips is deliberate and is not a missing measurement.
//     They measure one coordinate, `train_gtheta.py:652-653` zeroes both the
//     target and the weight of output 1 there, and the model is therefore never
//     supervised on it. Zero switches the term off. A table that carried the
//     old 43 um there would apply an unsupervised output.
//
//  3. A volume `classOf` does not know gets NO table entry, and the caller's
//     honest response is the same as `NoiseTable`'s: fall back to the constants
//     it was given rather than invent a sigma. `lookup` reports that by
//     returning false.
#pragma once

#include "LearnedNoise.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace collider_ml {

/// (sig0, sig1) per (class, pt bin, |eta| bin), in micrometres.
///
/// Binned identically to `NoiseTable` and for the same reason: it is the
/// binning `chi2_gate.measured_cov` keys on, so anything else would be
/// silently mis-indexed rather than loudly wrong.
class CellSigmaTable {
 public:
  struct Sigma {
    double s0 = 0.0;
    double s1 = 0.0;
  };

  /// Reads what `src/prop/export_cell_sigma.py` writes. Returns false and says
  /// why on stderr rather than throwing, so the caller can keep the constants
  /// and be loud about it.
  bool load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      std::fprintf(stderr, "CellSigmaTable: cannot open %s\n", path.c_str());
      return false;
    }
    std::int32_t magic = 0, nClass = 0, nPt = 0, nEta = 0;
    bool ok = std::fread(&magic, 4, 1, f) == 1 &&
              std::fread(&nClass, 4, 1, f) == 1 &&
              std::fread(&nPt, 4, 1, f) == 1 &&
              std::fread(&nEta, 4, 1, f) == 1;
    if (!ok || magic != kMagic) {
      std::fprintf(stderr, "CellSigmaTable: %s is not a cell sigma table\n",
                   path.c_str());
      std::fclose(f);
      return false;
    }
    if (nClass != kNClass || nPt != kNPt || nEta != kNEta) {
      std::fprintf(stderr,
                   "CellSigmaTable: %s is binned %dx%dx%d, this build expects "
                   "%dx%dx%d\n",
                   path.c_str(), nClass, nPt, nEta, kNClass, kNPt, kNEta);
      std::fclose(f);
      return false;
    }
    const std::size_t nCell = kNClass * kNPt * kNEta;
    m_cell.assign(nCell, Sigma{});
    m_have.assign(nCell, 0);
    m_fallback.assign(kNClass, Sigma{});
    double buf[2] = {0.0, 0.0};
    for (int c = 0; c < kNClass; ++c) {
      if (std::fread(buf, sizeof(double), 2, f) != 2) {
        ok = false;
        break;
      }
      m_fallback[c] = Sigma{buf[0], buf[1]};
    }
    for (std::size_t i = 0; ok && i < nCell; ++i) {
      std::uint8_t have = 0;
      if (std::fread(&have, 1, 1, f) != 1 ||
          std::fread(buf, sizeof(double), 2, f) != 2) {
        ok = false;
        break;
      }
      m_have[i] = have;
      m_cell[i] = Sigma{buf[0], buf[1]};
    }
    std::fclose(f);
    if (!ok) {
      std::fprintf(stderr, "CellSigmaTable: %s is truncated\n", path.c_str());
      m_cell.clear();
      return false;
    }
    m_loaded = true;
    return true;
  }

  bool loaded() const { return m_loaded; }

  /// The sigma for one jump, in micrometres. False leaves `*out` untouched.
  ///
  /// A cell the measurement never filled falls back to the class value, which
  /// is what `chi2_gate.measured_cov` does with its pt_bin = eta_bin = -1 row.
  /// Interpolating between neighbouring cells would be inventing a measurement.
  bool lookup(unsigned volume, double pt, double absEta, Sigma* out) const {
    ModuleClass cls{};
    if (!m_loaded || !classOf(volume, &cls)) {
      return false;
    }
    const int c = static_cast<int>(cls);
    const std::size_t i =
        (static_cast<std::size_t>(c) * kNPt + ptBin(pt)) * kNEta +
        etaBin(absEta);
    *out = (m_have[i] != 0 ? m_cell[i] : m_fallback[c]);
    return true;
  }

 private:
  static constexpr std::int32_t kMagic = 0x53474D41;  // "SGMA"
  bool m_loaded = false;
  std::vector<Sigma> m_cell;
  std::vector<std::uint8_t> m_have;
  std::vector<Sigma> m_fallback;
};

}  // namespace collider_ml
