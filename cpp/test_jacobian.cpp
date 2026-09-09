// Check cpp/LearnedJacobian.hpp against the Python it is a port of, and check
// the two structural facts ACTS relies on.
//
// `export_jacobian.py` runs `jacobian_free.py` -- the file whose correspondence
// with F_bound was verified to 3.9e-15 -- and writes both ends of it.
//
//   c++ -std=c++20 -O2 -I<eigen> test_jacobian.cpp -o test_jacobian
//   ./test_jacobian reference_jac.bin
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "LearnedJacobian.hpp"

namespace {

constexpr int kNIn = 12;
constexpr int kNOut = 74;

/// Error relative to the size of the QUANTITY, not of the entry.
///
/// Dividing by |want| per entry is the trap `jacobian_helix.rel()` documents: a
/// component that happens to be near zero reports a huge relative error for an
/// absolute error of nothing. D is far worse for this than the kernel outputs
/// were, because more than half its entries are structural zeros and its
/// non-zero entries span 1e-4 to 1e+4. So the scale is per JUMP, the largest
/// entry of the reference matrix, which is what a covariance transport actually
/// multiplies by.
struct Stat {
  const char* name;
  std::vector<double> rel;

  void add(double r) { rel.push_back(r); }

  double quantile(double q) {
    if (rel.empty()) {
      return 0.0;
    }
    auto n = static_cast<std::size_t>(q * static_cast<double>(rel.size() - 1));
    std::nth_element(rel.begin(), rel.begin() + n, rel.end());
    return rel[n];
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "reference_jac.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return 2;
  }
  std::int32_t n = 0, nin = 0, nout = 0;
  double mass = 0.0;
  f.read(reinterpret_cast<char*>(&n), 4);
  f.read(reinterpret_cast<char*>(&nin), 4);
  f.read(reinterpret_cast<char*>(&nout), 4);
  f.read(reinterpret_cast<char*>(&mass), 8);
  if (nin != kNIn || nout != kNOut) {
    std::fprintf(stderr, "layout mismatch: %d/%d\n", nin, nout);
    return 2;
  }
  std::vector<double> in(static_cast<std::size_t>(n) * nin);
  std::vector<double> want(static_cast<std::size_t>(n) * nout);
  f.read(reinterpret_cast<char*>(in.data()),
         static_cast<std::streamsize>(in.size() * sizeof(double)));
  f.read(reinterpret_cast<char*>(want.data()),
         static_cast<std::streamsize>(want.size() * sizeof(double)));

  Stat jac{"jacTransport D", {}};
  Stat der{"derivative t", {}};
  Stat pth{"path3d", {}};
  Stat tof{"time of flight", {}};
  // The two blocks EigenStepper.ipp:338-339 asserts. These are not a comparison
  // against Python; they are the structural claim that makes this matrix a
  // legal jacTransport at all, and they are checked on the C++ side because
  // that is the side that will be handed to ACTS.
  double worstTopLeft = 0.0, worstBottomLeft = 0.0;

  for (std::int32_t i = 0; i < n; ++i) {
    const double* x = &in[static_cast<std::size_t>(i) * nin];
    const double* y = &want[static_cast<std::size_t>(i) * nout];

    collider_ml::Jump j;
    j.pos = {x[0], x[1], x[2]};
    j.mom = {x[3], x[4], x[5]};
    j.q = x[6];
    const double bz = x[7];
    const double s = x[8];
    if (bz != collider_ml::detail::kBHelix) {
      std::fprintf(stderr,
                   "reference was made at bz=%g, kernel is fixed at %g\n", bz,
                   collider_ml::detail::kBHelix);
      return 2;
    }

    // The helix endpoint momentum comes from the reference, not from a solve
    // here. `jumpJacobian` is what is under test; re-deriving its argument
    // would fold the arc-length solver into the comparison and a disagreement
    // would not say which of the two was wrong.
    const collider_ml::Vec3 helixMom(x[9], x[10], x[11]);
    const collider_ml::JumpJacobian jj =
        collider_ml::jumpJacobian(j, helixMom, s, mass);

    double scale = 0.0;
    for (int k = 0; k < 64; ++k) {
      scale = std::max(scale, std::abs(y[k]));
    }
    if (scale <= 0.0) {
      scale = 1.0;
    }
    for (int r = 0; r < 8; ++r) {
      for (int c = 0; c < 8; ++c) {
        jac.add(std::abs(jj.D(r, c) - y[r * 8 + c]) / scale);
      }
    }

    double tscale = 0.0;
    for (int k = 0; k < 8; ++k) {
      tscale = std::max(tscale, std::abs(y[64 + k]));
    }
    for (int k = 0; k < 8; ++k) {
      der.add(std::abs(jj.t(k) - y[64 + k]) / (tscale > 0.0 ? tscale : 1.0));
    }

    pth.add(std::abs(jj.path3d - y[72]) / std::max(std::abs(y[72]), 1e-30));
    tof.add(std::abs(jj.dt - y[73]) / std::max(std::abs(y[73]), 1e-30));

    worstTopLeft = std::max(
        worstTopLeft,
        (jj.D.topLeftCorner<4, 4>() - Eigen::Matrix4d::Identity())
            .cwiseAbs()
            .maxCoeff());
    worstBottomLeft =
        std::max(worstBottomLeft, jj.D.bottomLeftCorner<4, 4>().cwiseAbs()
                                      .maxCoeff());
  }

  std::printf("%d jumps, mass %.6f GeV\n\n", n, mass);
  std::printf("%-18s %12s %12s %12s\n", "against Python", "median", "p99",
              "max");
  double worst = 0.0;
  for (Stat* st : {&jac, &der, &pth, &tof}) {
    const double mx = st->quantile(1.0);
    worst = std::max(worst, mx);
    std::printf("%-18s %12.2e %12.2e %12.2e\n", st->name, st->quantile(0.5),
                st->quantile(0.99), mx);
  }

  std::printf("\nthe blocks EigenStepper.ipp:338 asserts, on the C++ D:\n");
  std::printf("  topLeft 4x4 - I    max %.2e\n", worstTopLeft);
  std::printf("  bottomLeft 4x4     max %.2e\n", worstBottomLeft);

  // 1e-10 is the bar the Jacobian work in this project used against finite
  // differences, and the same bar applies to a port. The asserts are compiled
  // out under NDEBUG in a release ACTS, so they are checked here at the
  // tolerance Eigen's own isIdentity/isZero use rather than assumed.
  const bool blocksOk = worstTopLeft < 1e-12 && worstBottomLeft < 1e-12;
  const bool pass = worst < 1e-10 && blocksOk;
  std::printf("%s  (worst relative error %.2e)\n", pass ? "PASS" : "FAIL",
              worst);
  return pass ? 0 : 1;
}
