#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
inline void write_payload(unsigned char* data,std::size_t size,std::uint64_t id,char kind){
 if(size<16)throw std::invalid_argument("payload too short");
 std::uint64_t check=id^0xa67d31593b72e40cULL^std::uint64_t(static_cast<unsigned char>(kind));
 std::memcpy(data,&id,8);std::memcpy(data+8,&check,8);
}
inline bool valid_payload(const unsigned char* data,std::size_t size,std::uint64_t id,char kind){
 if(size<16)return false;std::uint64_t actual=0,check=0;std::memcpy(&actual,data,8);std::memcpy(&check,data+8,8);
 return actual==id&&check==(id^0xa67d31593b72e40cULL^std::uint64_t(static_cast<unsigned char>(kind)));
}
