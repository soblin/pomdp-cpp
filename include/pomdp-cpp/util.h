#include <memory>

namespace pomdp_cpp {

template <typename T> using ptr = std::shared_ptr<T>;

template <typename T, class... Args> ptr<T> make_ptr(Args... args) { return std::make_shared<T>(args...); }

template <typename T> ptr<T> make_ptr() { return std::make_shared<T>(); }

} // namespace pomdp_cpp
