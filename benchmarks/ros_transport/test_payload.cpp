#include "payload.hpp"
#include <array>
#include <cassert>
int main(){std::array<unsigned char,32> b{};write_payload(b.data(),b.size(),123,'d');assert(valid_payload(b.data(),b.size(),123,'d'));assert(!valid_payload(b.data(),b.size(),124,'d'));assert(!valid_payload(b.data(),b.size(),123,'r'));b[8]^=1;assert(!valid_payload(b.data(),b.size(),123,'d'));assert(!valid_payload(b.data(),8,123,'d'));}
