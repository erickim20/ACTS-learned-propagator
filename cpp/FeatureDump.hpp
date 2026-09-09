// The network's inputs as the CKF actually builds them, written out so they can
// be compared against the distribution the model was fitted on.
//
// WHY THIS EXISTS. `test_kernel.cpp` checks the C++ forward pass against 2,000
// reference vectors from the Python path, and both sides of that comparison are
// in the training units. It therefore cannot see an error in the code that
// BUILDS the input vector at run time, and one lived there: `LearnedStepper`'s
// `toJump` filled the field feature straight from `getField`, which answers in
// ACTS's native units, so a 2 T solenoid reached the network as 5.996e-4
// against a feature whose training mean is 1.907 and standard deviation 0.191.
// Every learned-on number taken in this repository before that was fixed was
// taken with one input pinned ten standard deviations outside its training
// range. Every other input has the same exposure and nothing was watching it.
//
// So this is not a debug aid. It is the acceptance check that stands between a
// unit error and a month of measuring noise, and `src/prop/feature_check.py` is
// the half of it that reads the file and states which inputs are inside their
// training distribution and which are not.
//
// What is recorded. The raw feature vector, at the point it enters `forward()`,
// before the (x - mu) / sd normalisation. Raw, because mu and sd are stored in
// raw units alongside the weights and the whole question is whether the two
// agree.
//
// Off unless `FEATURE_DUMP` names an output path, so a normal run pays one
// predictable branch per transport and writes nothing.
//
//     FEATURE_DUMP=/output/runs/x/features.bin
//     FEATURE_DUMP_MAX=500000        # rows kept, default 500,000
//
// The rows kept are the first `FEATURE_DUMP_MAX`, not a sample of the whole
// run. The intended use is a short run, and the file's own header carries both
// the rows kept and the rows offered so the reader can say which it got.
//
// FORMAT. Little-endian, and the same shape as `cpp/reference.bin`:
//
//     int32   rows        rows actually in the file
//     int32   cols        14, the input width the kernel was built with
//     int64   offered     forward() calls the run made, kept or not
//     float64 x[rows * cols]   row-major
//
// The count is patched into the header when the process exits. A run that dies
// before then leaves the rows behind with a zero count, and the reader falls
// back to the file size, which is why `cols` comes first and `rows` is allowed
// to be wrong.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace collider_ml {

class FeatureDump {
 public:
  static FeatureDump& instance() {
    static FeatureDump d;
    return d;
  }

  bool on() const { return m_on; }

  /// One feature vector, as `forward()` is about to see it.
  ///
  /// `n` is passed rather than assumed so that a kernel built with a different
  /// input width writes a file that says so instead of one that is silently
  /// misread. The lock is taken because the Sequencer's thread count is a
  /// config value and this must not depend on it being 1.
  void record(const double* x, int n) {
    if (!m_on) {
      return;
    }
    std::lock_guard<std::mutex> lock(m_mu);
    ++m_offered;
    if (m_rows >= m_max) {
      return;
    }
    if (m_f == nullptr && !open(n)) {
      return;
    }
    if (n != m_cols) {
      return;
    }
    if (std::fwrite(x, sizeof(double), static_cast<std::size_t>(n), m_f)
        == static_cast<std::size_t>(n)) {
      ++m_rows;
    }
  }

  ~FeatureDump() { close(); }

 private:
  FeatureDump() {
    const char* p = std::getenv("FEATURE_DUMP");
    if (p == nullptr || *p == '\0') {
      return;
    }
    m_path = p;
    m_on = true;
    if (const char* m = std::getenv("FEATURE_DUMP_MAX");
        m != nullptr && *m != '\0') {
      const long v = std::strtol(m, nullptr, 10);
      if (v > 0) {
        m_max = static_cast<std::int64_t>(v);
      }
    }
  }

  bool open(int n) {
    m_f = std::fopen(m_path.c_str(), "wb");
    if (m_f == nullptr) {
      std::fprintf(stderr, "[featdump] cannot write %s; dump off\n",
                   m_path.c_str());
      m_on = false;
      return false;
    }
    m_cols = n;
    const std::int32_t hdr[2] = {0, static_cast<std::int32_t>(n)};
    const std::int64_t offered = 0;
    std::fwrite(hdr, sizeof(std::int32_t), 2, m_f);
    std::fwrite(&offered, sizeof(std::int64_t), 1, m_f);
    return true;
  }

  void close() {
    if (m_f == nullptr) {
      return;
    }
    const std::int32_t rows = static_cast<std::int32_t>(m_rows);
    const std::int64_t offered = m_offered;
    if (std::fseek(m_f, 0, SEEK_SET) == 0) {
      std::fwrite(&rows, sizeof(std::int32_t), 1, m_f);
      std::fseek(m_f, sizeof(std::int32_t) * 2, SEEK_SET);
      std::fwrite(&offered, sizeof(std::int64_t), 1, m_f);
    }
    std::fclose(m_f);
    m_f = nullptr;
    std::fprintf(stderr,
                 "[featdump] %lld of %lld input vectors, %d wide -> %s\n",
                 static_cast<long long>(m_rows),
                 static_cast<long long>(m_offered), m_cols, m_path.c_str());
  }

  bool m_on = false;
  std::string m_path;
  std::FILE* m_f = nullptr;
  int m_cols = 0;
  std::int64_t m_rows = 0;
  std::int64_t m_offered = 0;
  std::int64_t m_max = 500000;
  std::mutex m_mu;
};

}  // namespace collider_ml
