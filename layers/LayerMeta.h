#pragma once

struct LayerMeta {
    float alpha;
    std::string name;
    size_t size;
    INeuronBuilder &neuronBuilder;
    float width;
    float zShift;
};