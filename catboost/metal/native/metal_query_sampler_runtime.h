#pragma once
#import <Metal/Metal.h>
#include "metal_sort.h"
#include "metal_query_sampler_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Resident query/document sampling. RNG priorities and whole-query masks are
// independent inputs so target-specific CUDA seed protocols can be connected.
class CBMQuerySamplerRuntime {
public:
    struct Shape { uint32_t Rows,Groups,Pairs,Reserved; };
    static uint64_t RequiredBytes(uint32_t rows,uint32_t groups) {
        if (!rows || rows>(1u<<24) || !groups || groups>rows)
            throw std::runtime_error("Invalid query sampler capacity");
        uint64_t scan=0,sort=12ull*rows;
        for(uint32_t n=(rows+255)/256;;n=(n+255)/256){scan+=4ull*n;if(n==1)break;}
        for(uint32_t n=16*((rows+255)/256);;n=(n+255)/256){sort+=4ull*n;if(n==1)break;}
        return 36ull*rows+28ull*groups+32+scan+sort;
    }
    CBMQuerySamplerRuntime(id<MTLDevice> device,uint32_t rows,uint32_t groups,const uint32_t* offsets,
        uint64_t budget=(1ull<<30)) : Device(device),Rows(rows),Groups(groups) {
        Require(device && rows && rows<=(1u<<24) && groups && groups<=rows && offsets && offsets[0]==0 && offsets[groups]==rows,
            "Invalid query sampler dimensions or offsets");
        const uint64_t expected=RequiredBytes(rows,groups);
        Require(budget<=(1ull<<30) && expected<=budget,"Query sampler exceeds its 1 GiB workspace budget");
        std::vector<uint32_t> qids(rows);
        for(uint32_t q=0;q<groups;++q) {
            Require(offsets[q]<offsets[q+1] && offsets[q+1]<=rows,"Query sampler offsets must increase strictly");
            std::fill(qids.begin()+offsets[q],qids.begin()+offsets[q+1],q);
        }
        OriginalOffsets=Buffer(offsets,4ull*(groups+1));OriginalQids=Buffer(qids.data(),4ull*rows);
        KeysA=Buffer(nullptr,4ull*rows);KeysB=Buffer(nullptr,4ull*rows);IndicesA=Buffer(nullptr,4ull*rows);IndicesB=Buffer(nullptr,4ull*rows);
        Mask=Buffer(nullptr,4ull*rows);RowPrefix=Buffer(nullptr,4ull*rows);
        Counts=Buffer(nullptr,4ull*groups);DocPrefix=Buffer(nullptr,4ull*groups);LivePrefix=Buffer(nullptr,4ull*groups);PairPrefix=Buffer(nullptr,4ull*groups);
        Documents=Buffer(nullptr,4ull*rows);QueryIds=Buffer(nullptr,4ull*rows);Offsets=Buffer(nullptr,4ull*(groups+1));PairOffsets=Buffer(nullptr,4ull*(groups+1));
        OutputShape=Buffer(nullptr,16);Status=Buffer(nullptr,4);
        for(uint32_t n=(rows+255)/256;;n=(n+255)/256){Upper.push_back(Buffer(nullptr,4ull*n));if(n==1)break;}
        Sort=std::make_unique<CBMSortU32Workspace>(device,rows);Bytes+=Sort->AllocatedBytes();
        Require(Bytes==expected,"Query sampler allocation accounting mismatch");
        Shaders=GetShaders(device);
    }
    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t CudaMaximumQuerySize() const { return std::min<uint32_t>(uint64_t(2)*Rows/Groups+8,1023); }
    id<MTLBuffer> SampledDocuments() const { return Documents; }
    id<MTLBuffer> SampledQueryIds() const { return QueryIds; }
    id<MTLBuffer> SampledOffsets() const { return Offsets; }
    id<MTLBuffer> SampledPairOffsets() const { return PairOffsets; }
    id<MTLBuffer> SampledMask() const { return Mask; }
    id<MTLBuffer> ShapeBuffer() const { return OutputShape; }

    void Encode(id<MTLCommandBuffer> command,id<MTLBuffer> randomKeys,id<MTLBuffer> queryMask,
        float fraction,uint32_t maxQuery=0,uint32_t pairLimit=(1u<<24),uint64_t* dispatches=nullptr) {
        Require(command && command.device==Device && command.status==MTLCommandBufferStatusNotEnqueued && command.retainedReferences,
            "Query sampler needs an uncommitted retained command on its device");
        Require(!LastCommand || LastCommand==command || LastCommand.status==MTLCommandBufferStatusCompleted || LastCommand.status==MTLCommandBufferStatusError,
            "Finish the previous sampler command before reusing its workspace");
        Require(std::isfinite(fraction) && fraction>0 && fraction<=1,"Query sampling fraction must be in (0, 1]");
        if(!maxQuery)maxQuery=CudaMaximumQuerySize();
        Require(maxQuery>=2 && maxQuery<=1023 && pairLimit<0xffffffffu,"Invalid sampler query or pair capacity");
        CheckBuffer(randomKeys,4ull*Rows);CheckBuffer(queryMask,4ull*Groups);
        LastCommand=command;Params p={Rows,Groups,maxQuery,pairLimit,fraction,0,0,0};
        auto clear=[command blitCommandEncoder];Require(clear!=nil,"Query sampler clear allocation failed");
        [clear fillBuffer:Status range:NSMakeRange(0,4) value:0];
        [clear copyFromBuffer:randomKeys sourceOffset:0 toBuffer:KeysA destinationOffset:0 size:4ull*Rows];[clear endEncoding];
        Dispatch(command,"InitializeQuerySamplerRows",{IndicesA},p,Rows,dispatches);
        Sort->Encode(command,KeysA,IndicesA,Rows,KeysB,IndicesB,dispatches);
        Dispatch(command,"GatherQuerySamplerKeys",{OriginalQids,IndicesB,KeysA},p,Rows,dispatches);
        // A stable query-ID pass turns the priority sort into (query,priority).
        Sort->Encode(command,KeysA,IndicesB,Rows,KeysB,IndicesA,dispatches);
        Dispatch(command,"CountQuerySamplerRows",{OriginalOffsets,queryMask,Counts,LivePrefix,PairPrefix,Status},p,Groups,dispatches);
        Dispatch(command,"MarkQuerySamplerRows",{IndicesA,OriginalQids,OriginalOffsets,Counts,Mask},p,Rows,dispatches);
        auto copy=[command blitCommandEncoder];Require(copy!=nil,"Query sampler prefix copy allocation failed");
        [copy copyFromBuffer:Mask sourceOffset:0 toBuffer:RowPrefix destinationOffset:0 size:4ull*Rows];
        [copy copyFromBuffer:Counts sourceOffset:0 toBuffer:DocPrefix destinationOffset:0 size:4ull*Groups];[copy endEncoding];
        Scan(command,RowPrefix,Rows,1u<<24,dispatches);Scan(command,DocPrefix,Groups,1u<<24,dispatches);
        Scan(command,LivePrefix,Groups,1u<<24,dispatches);Scan(command,PairPrefix,Groups,pairLimit,dispatches);
        Dispatch(command,"ScatterQuerySamplerRows",{Mask,RowPrefix,OriginalQids,LivePrefix,Documents,QueryIds},p,Rows,dispatches);
        Dispatch(command,"ScatterQuerySamplerGroups",{Counts,DocPrefix,LivePrefix,PairPrefix,Offsets,PairOffsets},p,Groups,dispatches);
        Dispatch(command,"FinalizeQuerySamplerShape",{RowPrefix,DocPrefix,LivePrefix,PairPrefix,Offsets,PairOffsets,OutputShape,Status},p,1,dispatches);
    }
    Shape ReadShape() const {
        Require(LastCommand && LastCommand.status==MTLCommandBufferStatusCompleted,"Complete the sampler command before reading its shape");
        const uint32_t status=*static_cast<const uint32_t*>(Status.contents);
        Require(!(status&4),"Sampled query pairs exceed the configured capacity");
        Require(!status,"Invalid query sampler mask or prefix state (status "+std::to_string(status)+")");
        const auto shape=*static_cast<const Shape*>(OutputShape.contents);
        Require(shape.Rows<=Rows && shape.Groups<=Groups,"Invalid compacted query shape");return shape;
    }
private:
    struct Params { uint32_t Rows,Groups,MaxQuery,PairLimit;float Fraction;uint32_t R0,R1,R2; };
    struct ShaderSet { std::unordered_map<std::string,id<MTLComputePipelineState>> Pipelines; };
    id<MTLDevice> Device;id<MTLCommandBuffer> LastCommand;uint32_t Rows,Groups;uint64_t Bytes=0;
    id<MTLBuffer> OriginalOffsets,OriginalQids,KeysA,KeysB,IndicesA,IndicesB,Mask,RowPrefix,Counts,DocPrefix,LivePrefix,PairPrefix;
    id<MTLBuffer> Documents,QueryIds,Offsets,PairOffsets,OutputShape,Status;
    std::vector<id<MTLBuffer>> Upper;std::unique_ptr<CBMSortU32Workspace> Sort;std::shared_ptr<ShaderSet> Shaders;
    static void Require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
    // Literal validation messages allocate only when a check fails.
    static void Require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static std::shared_ptr<ShaderSet> GetShaders(id<MTLDevice> device) {
        static std::mutex mutex;static std::unordered_map<uint64_t,std::shared_ptr<ShaderSet>> cache;
        std::lock_guard<std::mutex> guard(mutex);auto& result=cache[device.registryID];if(result)return result;
        auto built=std::make_shared<ShaderSet>();NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];
        if(@available(macOS 13.0,*))options.languageVersion=MTLLanguageVersion3_0;else throw std::runtime_error("Query sampler requires macOS 13+");
        options.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalQuerySamplerSource] options:options error:&error];
        Require(library!=nil,error ? error.localizedDescription.UTF8String : "Query sampler compilation failed");
        for(const char* name:{"InitializeQuerySamplerRows","GatherQuerySamplerKeys","CountQuerySamplerRows","MarkQuerySamplerRows",
            "ScanQuerySamplerPrefix","AddQuerySamplerPrefix","ScatterQuerySamplerRows","ScatterQuerySamplerGroups","FinalizeQuerySamplerShape"}) {
            auto fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(fn!=nil,std::string("Missing sampler kernel ")+name);
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
            Require(pipeline && pipeline.maxTotalThreadsPerThreadgroup>=256 && pipeline.threadExecutionWidth>=8 && pipeline.threadExecutionWidth<=128 && 256%pipeline.threadExecutionWidth==0,
                error ? error.localizedDescription.UTF8String : "Unsupported sampler threadgroup");built->Pipelines.emplace(name,pipeline);
        }
        result=built;return result;
    }
    id<MTLBuffer> Buffer(const void* source,uint64_t size) {
        Require(size && size<=Device.maxBufferLength,"Query sampler buffer exceeds device limits");
        auto result=source ? [Device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
            :[Device newBufferWithLength:size options:MTLResourceStorageModeShared];
        Require(result!=nil,"Query sampler buffer allocation failed");if(!source)std::memset(result.contents,0,size);Bytes+=size;return result;
    }
    void CheckBuffer(id<MTLBuffer> buffer,uint64_t size) const {
        Require(buffer && buffer.device==Device && buffer.length>=size,"Missing or undersized query sampler input buffer");
    }
    void Dispatch(id<MTLCommandBuffer> command,const char* name,std::initializer_list<id<MTLBuffer>> buffers,
        const Params& p,uint32_t count,uint64_t* dispatches) {
        auto encoder=[command computeCommandEncoder];Require(encoder!=nil,"Query sampler encoder allocation failed");
        [encoder setComputePipelineState:Shaders->Pipelines.at(name)];NSUInteger index=0;
        for(auto buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];[encoder setBytes:&p length:sizeof(p) atIndex:index];
        [encoder dispatchThreadgroups:MTLSizeMake((count+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        if(dispatches)++*dispatches;
    }
    void Scan(id<MTLCommandBuffer> command,id<MTLBuffer> input,uint32_t elements,uint32_t limit,uint64_t* dispatches) {
        std::vector<id<MTLBuffer>> levels={input};std::vector<uint32_t> sizes={elements};
        for(uint32_t n=elements;n>1;n=(n+255)/256) {
            const auto next=Upper[levels.size()-1];
            Dispatch(command,"ScanQuerySamplerPrefix",{levels.back(),next},Params{n,0,0,limit,0,0,0,0},n,dispatches);
            levels.push_back(next);sizes.push_back((n+255)/256);
        }
        for(size_t level=levels.size();level>2;--level) {
            const uint32_t index=level-3;
            Dispatch(command,"AddQuerySamplerPrefix",{levels[index],levels[index+1]},Params{sizes[index],0,0,limit,0,0,0,0},sizes[index],dispatches);
        }
    }
};
