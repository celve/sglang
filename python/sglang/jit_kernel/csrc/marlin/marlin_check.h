#pragma once
#include <stdexcept>
#include <sstream>

// Standalone replacement for TORCH_CHECK.
// Supports variadic args: MARLIN_CHECK(cond, "msg", arg1, arg2, ...)
namespace marlin_check_detail {
template <typename... Args>
[[noreturn]] inline void fail(const char* expr, Args&&... args) {
    std::ostringstream os;
    os << "Marlin check failed: " << expr;
    if constexpr (sizeof...(args) > 0) {
        os << " -- ";
        (os << ... << std::forward<Args>(args));
    }
    throw std::runtime_error(os.str());
}
}  // namespace marlin_check_detail

#define MARLIN_CHECK(cond, ...)                                            \
  do {                                                                     \
    if (!(cond)) ::marlin_check_detail::fail(#cond, ##__VA_ARGS__);        \
  } while (0)
