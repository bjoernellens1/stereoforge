#include <stereoforge/daxa_foundation.hpp>
#include <iostream>
#include <stdexcept>
#include <string_view>
int main(int argc, char** argv) {
    try {
        bool validation = false, benchmark = false;
        for(int i=1;i<argc;++i) {
            if(std::string_view(argv[i])=="--validation") validation=true;
            else if(std::string_view(argv[i])=="--benchmark") benchmark=true;
            else throw std::runtime_error("usage: daxa_foundation_contracts --validation [--benchmark]");
        }
        if (!validation) throw std::runtime_error("foundation contracts require --validation");
        stereoforge::DaxaFoundation context(validation);
        stereoforge::run_daxa_foundation(context, benchmark);
        context.close();
        if(context.validation_errors.load() || context.validation_warnings.load()) throw std::runtime_error("Khronos validation warnings/errors observed, including teardown");
        std::cout << "Daxa foundation: changing frame IDs 1,2,3 verified by GPU consumer; 12-byte final readback\n";
        return 0;
    } catch(std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
