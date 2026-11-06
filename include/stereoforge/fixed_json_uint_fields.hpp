#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
namespace stereoforge {
// Setup-time JSON offsets, allocation-free exact uint64 updates in retained
// storage.
template <size_t N> class FixedJsonUintFields {
  std::string template_;
  std::array<size_t, N> offsets_{};
  static void require(bool ok, const char *why) {
    if (!ok)
      throw std::invalid_argument(why);
  }

public:
  static constexpr std::string_view placeholder() {
    return "00000000000000000000";
  }
  FixedJsonUintFields(std::string value,
                      const std::array<std::string_view, N> &names)
      : template_(std::move(value)) {
    for (size_t i = 0; i < N; ++i) {
      const auto prefix = "\"" + std::string(names[i]) + "\":\"";
      const auto token = prefix + std::string(placeholder()) + "\"";
      const auto at = template_.find(token);
      require(at != std::string::npos &&
                  template_.find(token, at + 1) == std::string::npos,
              "missing or duplicate JSON uint placeholder");
      offsets_[i] = at + prefix.size();
      for (size_t j = 0; j < i; ++j)
        require(offsets_[j] != offsets_[i], "duplicate JSON uint field");
    }
  }
  std::string_view template_bytes() const { return template_; }
  void stamp(std::span<char> output,
             const std::array<uint64_t, N> &values) const {
    require(output.size() == template_.size(),
            "JSON uint template extent mismatch");
    for (size_t i = 0; i < N; ++i) {
      std::array<char, 20> digits{};
      auto result = std::to_chars(digits.data(), digits.data() + digits.size(),
                                  values[i]);
      require(result.ec == std::errc{}, "JSON uint formatting failed");
      auto field = output.subspan(offsets_[i], 20);
      std::fill(field.begin(), field.end(), '0');
      std::copy(digits.data(), result.ptr,
                field.end() - (result.ptr - digits.data()));
    }
  }
};
} // namespace stereoforge
