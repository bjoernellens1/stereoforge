// Scheduler-only native regression: no command submission or GPU dispatch.
#include <stereoforge/daxa_foundation.hpp>
#include <daxa/utils/task_graph.hpp>
#include <array>
#include <iostream>
int main(){try{
    stereoforge::DaxaFoundation context(true);auto& d=context.device;
    auto parameters=d.create_buffer({.size=4,.name="shared read-only parameters"});
    auto output=d.create_buffer({.size=4,.name="first independent clear"});
    auto layer=d.create_buffer({.size=4,.name="second independent clear"});
    {
        daxa::ExternalTaskBuffer ep({.buffer=parameters,.name="parameters"}),eo({.buffer=output,.name="output"}),el({.buffer=layer,.name="layer"});
        daxa::TaskGraph graph({.device=d,.reorder_tasks=true,.optimize_transient_lifetimes=true,.name="first access group batch regression"});
        auto p=graph.register_buffer(ep),o=graph.register_buffer(eo),l=graph.register_buffer(el);
        graph.add_task(daxa::InlineTask::Compute("first independent clear").reads(p).writes(o).executes([](daxa::TaskInterface){}));
        graph.add_task(daxa::InlineTask::Compute("second independent clear").reads(p).writes(l).executes([](daxa::TaskInterface){}));
        graph.add_task(daxa::InlineTask::Compute("dependent resolve").reads(l).reads_writes(o).executes([](daxa::TaskInterface){}));
        graph.submit({});graph.complete({}); // Formerly leaves batch zero empty and asserts.
    }
    d.destroy_buffer(parameters);d.destroy_buffer(output);d.destroy_buffer(layer);d.collect_garbage();context.close();
    if(context.validation_errors||context.validation_warnings)return 1;
    std::cout<<"Scheduler-only compact-batch regression passed (no GPU dispatch)\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
