// SDK-free diagnostic CPU serialization. No device synchronization or conversion.
#pragma once
#include <json.hpp>
#include <openssl/evp.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace sf::zed_io {
namespace fs = std::filesystem;
using Json = nlohmann::json;
struct SafeError : std::runtime_error { using std::runtime_error::runtime_error; };
inline void require(bool condition, const char* message) { if (!condition) throw SafeError(message); }
inline size_t checked_product(size_t a, size_t b) {
    require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "byte-count overflow");
    return a * b;
}

// EVP avoids deprecated OpenSSL 3 SHA interfaces. Payloads are hashed as they are written.
class Digest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx_{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
    Digest() { require(ctx_ && EVP_DigestInit_ex(ctx_.get(), EVP_sha256(), nullptr) == 1, "SHA256 initialization failed"); }
    void update(const void* data, size_t size) {
        require(EVP_DigestUpdate(ctx_.get(), data, size) == 1, "SHA256 update failed");
    }
    std::string finish() {
        std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
        unsigned int count = 0;
        require(EVP_DigestFinal_ex(ctx_.get(), bytes.data(), &count) == 1 && count == 32, "SHA256 finalization failed");
        std::ostringstream out;
        for (unsigned int i = 0; i < count; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
        return out.str();
    }
};
inline std::string hash_text(const std::string& text) {
    Digest digest; digest.update(text.data(), text.size()); return digest.finish();
}

class Writer {
    std::ofstream stream_;
    Digest digest_;
    std::uint64_t size_ = 0;
    std::string relative_;
public:
    Writer(const fs::path& root, std::string relative) : relative_(std::move(relative)) {
        // All callers supply generated single-component names under a freshly created root.
        require(fs::path(relative_).filename() == fs::path(relative_), "invalid generated payload path");
        require(!fs::exists(fs::symlink_status(root / relative_)), "payload already exists");
        stream_.exceptions(std::ios::badbit | std::ios::failbit);
        stream_.open(root / relative_, std::ios::binary | std::ios::out);
    }
    void write(const void* data, size_t count) {
        require(count <= size_t(std::numeric_limits<std::streamsize>::max()), "stream write size overflow");
        require(count <= std::numeric_limits<std::uint64_t>::max() - size_, "payload size overflow");
        stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        digest_.update(data, count); size_ += count;
    }
    Json finish() {
        stream_.flush(); stream_.close();
        return {{"path", relative_}, {"sha256", digest_.finish()}, {"size_bytes", size_}};
    }
};
inline constexpr size_t copy_chunk_bytes = 1024 * 1024;
template<class Sink>
inline void copy_stream(std::istream& input, Sink& writer) {
    require(input.good(), "input payload not readable");
    std::array<char, copy_chunk_bytes> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) writer.write(buffer.data(), static_cast<size_t>(count));
        require(!input.fail() || input.eof(), "input payload read failed");
    }
}
inline Json copy_and_hash(const fs::path& source, const fs::path& root, const std::string& name) {
    std::ifstream input(source, std::ios::binary);
    require(input.is_open(), "input payload could not be opened");
    input.exceptions(std::ios::badbit);
    Writer writer(root, name);
    copy_stream(input, writer);
    return writer.finish();
}
// The caller supplies accessible source extent; validate all arithmetic before touching it.
template<class Sink>
inline void write_rows(Sink& writer, const void* data, size_t extent,
                       size_t row_bytes, size_t height, size_t stride) {
    require(stride >= row_bytes, "row stride shorter than logical row");
    const size_t packed_bytes = checked_product(row_bytes, height);
    const size_t offset = checked_product(stride, height ? height - 1 : 0);
    require(row_bytes <= std::numeric_limits<size_t>::max() - offset, "row extent overflow");
    require(height == 0 || offset + row_bytes <= extent, "source extent too short");
    require(packed_bytes == 0 || data != nullptr, "null source buffer");
    if (packed_bytes == 0) return;
    const auto* base = static_cast<const std::uint8_t*>(data);
    if (stride == row_bytes) writer.write(base, packed_bytes);
    else for (size_t y = 0; y < height; ++y) writer.write(base + y * stride, row_bytes);
}
} // namespace sf::zed_io
