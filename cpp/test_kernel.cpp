// Check the C++ kernel against the Python that produced the report's numbers.
//
// `export_kernel.py` runs the production path -- closed_loop.predict, the same
// function every chained result came from -- on N jumps drawn from the teacher
// and writes both ends of it. This reads that file and compares. A port that
// is "obviously the same formulas" is exactly the kind of thing that is
// silently 1% wrong, and 1% of a 19 um residual is the whole learned effect.
//
//   c++ -std=c++20 -O2 -I$(brew --prefix eigen)/include/eigen3 \
//       test_kernel.cpp -o test_kernel && ./test_kernel
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "LearnedTransport.hpp"

namespace {

constexpr int kNIn = 22;
constexpr int kNOut = 15;

/// Error relative to the size of the QUANTITY, not of the entry.
///
/// Dividing by |want| per entry is the trap `jacobian_helix.rel()` documents:
/// a component that happens to be near zero reports a huge relative error for
/// an absolute error of nothing, and the headline becomes a property of the
/// sample rather than of the port. Scale by the RMS of the reference values
/// instead, which is the same rule that file settled on.
struct Stat {
  const char* name;
  std::vector<double> got, want;

  void add(double g, double w) {
    got.push_back(g);
    want.push_back(w);
  }

  double scale() const {
    double s2 = 0.0;
    for (double v : want) {
      s2 += v * v;
    }
    return want.empty() ? 1.0 : std::sqrt(s2 / static_cast<double>(want.size()));
  }

  std::vector<double> rel() const {
    const double s = scale();
    std::vector<double> r(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      r[i] = std::abs(got[i] - want[i]) / (s > 0.0 ? s : 1.0);
    }
    return r;
  }

  double quantile(double q) const {
    std::vector<double> v = rel();
    if (v.empty()) {
      return 0.0;
    }
    auto n = static_cast<std::size_t>(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "reference.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return 2;
  }
  std::int32_t n = 0;
  std::int32_t nin = 0;
  std::int32_t nout = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  f.read(reinterpret_cast<char*>(&nin), 4);
  f.read(reinterpret_cast<char*>(&nout), 4);
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

  Stat pos{"position   (mm)", {}, {}};
  Stat mom{"momentum   (GeV)", {}, {}};
  Stat arc{"arc length (mm)", {}, {}};
  Stat raw{"raw outputs", {}, {}};
  Stat sig{"sigma scale m", {}, {}};
  int okMismatch = 0;
  int skipped = 0;

  for (std::int32_t i = 0; i < n; ++i) {
    const double* x = &in[static_cast<std::size_t>(i) * nin];
    const double* y = &want[static_cast<std::size_t>(i) * nout];

    collider_ml::Jump j;
    j.pos = {x[0], x[1], x[2]};
    j.mom = {x[3], x[4], x[5]};
    j.q = x[6];
    j.c = {x[7], x[8], x[9]};
    j.n = {x[10], x[11], x[12]};
    j.e0 = {x[13], x[14], x[15]};
    j.e1 = {x[16], x[17], x[18]};
    j.bzMap = x[19];
    j.sig0 = x[20];
    j.sig1 = x[21];

    const collider_ml::Prediction p = collider_ml::transport(j);

    if (p.ok != (y[7] != 0.0)) {
      ++okMismatch;
    }
    if (y[7] == 0.0) {  // the helix never reaches; nothing downstream is defined
      ++skipped;
      continue;
    }

    for (int k = 0; k < 3; ++k) {
      pos.add(p.pos[k], y[k]);
      mom.add(p.mom[k], y[3 + k]);
    }
    arc.add(p.s, y[6]);
    sig.add(p.m, y[8]);
    for (int k = 0; k < 6; ++k) {
      raw.add(p.raw[k], y[9 + k]);
    }
  }

  std::printf("%d jumps  (%d with no intersection, skipped)\n\n", n, skipped);
  std::printf("%-18s %12s %12s %12s %12s\n", "", "median", "p99", "max",
              "RMS scale");
  double worst = 0.0;
  for (const Stat* s : {&pos, &mom, &arc, &raw, &sig}) {
    const double mx = s->quantile(1.0);
    worst = std::max(worst, mx);
    std::printf("%-18s %12.2e %12.2e %12.2e %12.3g\n", s->name,
                s->quantile(0.5), s->quantile(0.99), mx, s->scale());
  }
  std::printf("\nok-flag mismatches: %d\n", okMismatch);

  // 1e-10 is the tolerance the Jacobian work in this project used against
  // finite differences; the same bar applies to a port.
  const bool pass = worst < 1e-10 && okMismatch == 0;
  std::printf("%s  (worst relative error %.2e)\n", pass ? "PASS" : "FAIL",
              worst);
  return pass ? 0 : 1;
}
