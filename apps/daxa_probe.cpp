#include <stereoforge/daxa_foundation.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
int main(int argc, char** argv) {
    std::string output;
    try {
        bool validation = false;
        for(int i=1;i<argc;++i) {
            std::string_view arg(argv[i]);
            if(arg == "--validation") validation = true;
            else if(arg == "--output" && i+1<argc) output=argv[++i];
            else throw std::runtime_error("usage: daxa_probe [--validation] --output FILE");
        }
        if(output.empty()) throw std::runtime_error("--output FILE is required");
        stereoforge::DaxaFoundation context(validation);
        auto report=context.report();
        context.close();
        if(context.validation_errors.load() || context.validation_warnings.load()) throw std::runtime_error("Khronos validation warnings/errors observed, including teardown");
        std::ofstream file(output);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << report;
        return 0;
    } catch(std::exception const& e) {
        std::cerr << e.what() << '\n';
        if(!output.empty()) {
            std::ofstream file(output);
            file << "{\"status\":\"blocked\",\"daxa_commit\":\"" SF_DAXA_REVISION "\",\"slang_version\":\"" SF_DAXA_SLANG_VERSION "\",\"error\":" << std::quoted(e.what()) << "}\n";
        }
        return 1;
    }
}
