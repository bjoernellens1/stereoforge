// Compile independently of the optional SDK target; exercises production helpers.
#include "../benchmarks/zed_comparison/export_io.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
using namespace sf::zed_io;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
struct Counter {
    size_t calls = 0, bytes = 0, largest = 0;
    void write(const void*, size_t count) { ++calls; bytes += count; largest = std::max(largest, count); }
};
std::vector<std::uint8_t> read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
template<class F> void rejects(F operation) {
    bool failed = false;
    try { operation(); } catch (const std::exception&) { failed = true; }
    require(failed, "expected rejection");
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "supply fresh output directory");
        const fs::path root(argv[1]); require(fs::create_directory(root), "fresh output required");
        constexpr size_t width = 1280, height = 720, row = width * 4;
        std::vector<std::uint8_t> packed(row * height), padded((row + 13) * height, 0xed);
        for (size_t i=0; i<packed.size(); ++i) packed[i] = static_cast<std::uint8_t>((i * 17 + i / 257) % 251);
        Json report = {{"boundary", "CPU diagnostic serialization only; no SDK/GPU/FPS qualification"}, {"runs", Json::array()}};
        for (const auto* kind : {"bgra", "f32"}) {
            if (std::string(kind) == "f32") {
                constexpr std::uint32_t bits[]{0x7fc12345, 0x7f800000, 0xff800000, 0x3f800000, 0x80000000};
                for (size_t i=0; i<packed.size()/4; ++i) std::memcpy(packed.data()+i*4, &bits[i%5], 4);
            }
            for (size_t y=0; y<height; ++y) std::memcpy(padded.data()+y*(row+13), packed.data()+y*row, row);
            const std::string expected_sha = std::string(kind) == "bgra"
                ? "d86d95950673c473894109995d4df4b4369746f6af7388a4d468675269c9a2a6"
                : "9d923413201d25f34cdf2b76ee7c380ac76c6ea44bbd07327340e89c9e827655";
            for (bool padding : {false, true}) {
                const auto& bytes = padding ? padded : packed; const size_t stride = padding ? row+13 : row;
                Counter counter; write_rows(counter, bytes.data(), bytes.size(), row, height, stride);
                require(counter.calls == (padding ? height : 1) && counter.bytes == packed.size(), "write call count/bytes mismatch");
                for (int run=0; run<3; ++run) {
                    const std::string name = std::string(kind)+(padding ? "-padded-" : "-packed-")+std::to_string(run);
                    const auto start = Clock::now(); Writer writer(root, name);
                    write_rows(writer, bytes.data(), bytes.size(), row, height, stride);
                    const Json meta = writer.finish(); const double write_hash_ms = ms(start);
                    require(read(root/name) == packed, "logical byte mismatch");
                    require(meta.at("sha256") == expected_sha && meta.at("size_bytes") == packed.size(), "digest/size mismatch");
                    const auto hash_start = Clock::now(); Digest hash;
                    if (padding) for (size_t y=0; y<height; ++y) hash.update(bytes.data()+y*stride, row);
                    else hash.update(bytes.data(), bytes.size());
                    require(hash.finish() == expected_sha, "hash-only mismatch"); const double hash_ms = ms(hash_start);
                    report["runs"].push_back({{"kind", kind}, {"padded", padding}, {"run", run}, {"write_calls", counter.calls},
                        {"bytes", counter.bytes}, {"sha256", expected_sha}, {"write_plus_sha_ms", write_hash_ms}, {"sha_only_ms", hash_ms}});
                }
            }
        }
        Counter counter;
        rejects([&]{ write_rows(counter, packed.data(), packed.size()-1, row, height, row); });
        rejects([&]{ write_rows(counter, packed.data(), packed.size(), row, height, row-1); });
        rejects([&]{ write_rows(counter, packed.data(), packed.size(), row, height, std::numeric_limits<size_t>::max()); });
        rejects([&]{ checked_product(std::numeric_limits<size_t>::max(), 2); });
        rejects([&]{ write_rows(counter, nullptr, packed.size(), row, height, row); });
        rejects([&]{ Writer exists(root, "bgra-packed-0"); });
        rejects([&]{ Writer bad(root / "missing", "file"); });
        rejects([&]{ copy_and_hash(root/"absent", root, "unused"); });
        // A bounded generated stream on disk; never allocate the 32 MiB source in RAM.
        const auto source = root/"input.bin"; Digest source_hash;
        { std::ofstream out(source, std::ios::binary); std::array<char, 65536> chunk{};
          for (size_t i=0; i<chunk.size(); ++i) chunk[i]=static_cast<char>(i%239);
          for (int i=0; i<512; ++i) { out.write(chunk.data(), chunk.size()); source_hash.update(chunk.data(), chunk.size()); } }
        const auto copy_start = Clock::now(); const auto copied = copy_and_hash(source, root, "copy.bin"); const double copy_ms=ms(copy_start);
        require(copied.at("size_bytes") == 32*1024*1024 && copied.at("sha256") == "d34d2110e31c97afa551b8faeb889cb6b72a52dd5ed7391d2c7f6367b2a10558"
            && copied.at("sha256") == source_hash.finish(), "copy metadata mismatch");
        { std::ifstream input(source, std::ios::binary); Counter chunks; copy_stream(input, chunks);
          require(chunks.calls == 32 && chunks.largest <= 1024*1024 && chunks.bytes == 32*1024*1024, "copy not bounded");
          report["copy"] = {{"bytes", chunks.bytes}, {"calls", chunks.calls}, {"maximum_chunk_bytes", chunks.largest}, {"write_plus_sha_ms", copy_ms}, {"sha256", copied.at("sha256")}}; }
        { std::ifstream a(source, std::ios::binary), b(root/"copy.bin", std::ios::binary); std::array<char,65536> x{}, y{};
          while(a) { a.read(x.data(),x.size()); b.read(y.data(),y.size()); require(a.gcount()==b.gcount() && x==y, "source copy byte mismatch"); } }
        rejects([&]{ copy_and_hash(source, root, "copy.bin"); });
        std::ifstream failed(source, std::ios::binary); failed.setstate(std::ios::failbit);
        rejects([&]{ copy_stream(failed, counter); });
        report["byte_parity"] = "pass"; report["rejection_cases"] = "pass";
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
