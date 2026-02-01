#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if ((ret) != ACL_SUCCESS) {                                                            \
            std::cerr << "ACL error " << __FILE__ << ":" << __LINE__ << " ret=" << (ret)       \
                      << std::endl;                                                            \
            return (ret);                                                                       \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                         \
    do {                                                                                       \
        if ((ret) != HCCL_SUCCESS) {                                                           \
            std::cerr << "HCCL error " << __FILE__ << ":" << __LINE__ << " ret=" << (ret)      \
                      << std::endl;                                                            \
            return (ret);                                                                       \
        }                                                                                      \
    } while (0)

struct Options {
    int rank = 0;
    int world = 2;
    int device = 0;
    std::string rootInfoPath = "rootinfo.bin";
    int warmup = 3;
    int iters = 10;
    uint64_t bytes = 1ULL << 30;  // 1 GiB
};

static bool FileExists(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static int WriteRootInfo(const std::string &path, const HcclRootInfo &info)
{
    std::cout << "Writing rootInfo to " << path << std::endl;
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) {
        std::cerr << "Failed to open rootInfo file for write: " << path << std::endl;
        return -1;
    }
    ofs.write(reinterpret_cast<const char *>(&info), sizeof(HcclRootInfo));
    if (!ofs.good()) {
        std::cerr << "Failed to write rootInfo file: " << path << std::endl;
        return -1;
    }
    return 0;
}

static int ReadRootInfo(const std::string &path, HcclRootInfo &info, int timeoutSec)
{
    const int sleepMs = 200;
    int waitedMs = 0;
    while (!FileExists(path)) {
        if (waitedMs >= timeoutSec * 1000) {
            std::cerr << "Timeout waiting for rootInfo file: " << path << std::endl;
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }

    std::cout << "Found rootInfo file: " << path << ", reading..." << std::endl;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) {
        std::cerr << "Failed to open rootInfo file for read: " << path << std::endl;
        return -1;
    }
    ifs.read(reinterpret_cast<char *>(&info), sizeof(HcclRootInfo));
    if (!ifs.good()) {
        std::cerr << "Failed to read rootInfo file: " << path << std::endl;
        return -1;
    }
    return 0;
}

static void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " --rank <0|1> --world 2 --device <id> "
              << "--root-info <path> [--bytes <n>] [--warmup <n>] [--iters <n>]\n";
}

static bool ParseArgs(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needValue = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--rank") {
            const char *v = needValue("--rank");
            if (!v) return false;
            opt.rank = std::atoi(v);
        } else if (key == "--world") {
            const char *v = needValue("--world");
            if (!v) return false;
            opt.world = std::atoi(v);
        } else if (key == "--device") {
            const char *v = needValue("--device");
            if (!v) return false;
            opt.device = std::atoi(v);
        } else if (key == "--root-info") {
            const char *v = needValue("--root-info");
            if (!v) return false;
            opt.rootInfoPath = v;
        } else if (key == "--bytes") {
            const char *v = needValue("--bytes");
            if (!v) return false;
            opt.bytes = std::strtoull(v, nullptr, 10);
        } else if (key == "--warmup") {
            const char *v = needValue("--warmup");
            if (!v) return false;
            opt.warmup = std::atoi(v);
        } else if (key == "--iters") {
            const char *v = needValue("--iters");
            if (!v) return false;
            opt.iters = std::atoi(v);
        } else {
            std::cerr << "Unknown arg: " << key << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseArgs(argc, argv, opt)) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (opt.world != 2) {
        std::cerr << "This test expects --world 2" << std::endl;
        return 1;
    }
    if (opt.rank != 0 && opt.rank != 1) {
        std::cerr << "This test expects --rank 0 or 1" << std::endl;
        return 1;
    }

    std::cout << "rank=" << opt.rank << " device=" << opt.device << " world=" << opt.world
              << " bytes=" << opt.bytes << std::endl;
    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(opt.device));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    HcclRootInfo rootInfo;
    if (opt.rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
        if (WriteRootInfo(opt.rootInfoPath, rootInfo) != 0) {
            return 1;
        }
        std::cout << "Rank0 wrote rootInfo to " << opt.rootInfoPath << std::endl;
    } else {
        if (ReadRootInfo(opt.rootInfoPath, rootInfo, 300) != 0) {
            return 1;
        }
        std::cout << "Rank1 read rootInfo from " << opt.rootInfoPath << std::endl;
    }

    std::cout << "rank=" << opt.rank << " before HcclCommInitRootInfo" << std::endl;
    HcclComm comm;
    HCCLCHECK(HcclCommInitRootInfo(opt.world, &rootInfo, opt.rank, &comm));
    std::cout << "rank=" << opt.rank << " after HcclCommInitRootInfo" << std::endl;

    const uint64_t count = opt.bytes / sizeof(float);
    const uint64_t bytes = count * sizeof(float);
    void *devBuf = nullptr;
    ACLCHECK(aclrtMalloc(&devBuf, bytes, ACL_MEM_MALLOC_HUGE_ONLY));

    void *hostBuf = nullptr;
    if (opt.rank == 0) {
        ACLCHECK(aclrtMallocHost(&hostBuf, bytes));
        std::memset(hostBuf, 0, static_cast<size_t>(bytes));
        ACLCHECK(aclrtMemcpy(devBuf, bytes, hostBuf, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    for (int i = 0; i < opt.warmup; ++i) {
        std::cout << "rank=" << opt.rank << " warmup iter=" << i << " before Send/Recv" << std::endl;
        if (opt.rank == 0) {
            HCCLCHECK(HcclSend(devBuf, count, HCCL_DATA_TYPE_FP32, 1, comm, stream));
        } else {
            HCCLCHECK(HcclRecv(devBuf, count, HCCL_DATA_TYPE_FP32, 0, comm, stream));
        }
        ACLCHECK(aclrtSynchronizeStream(stream));
        std::cout << "rank=" << opt.rank << " warmup iter=" << i << " after Send/Recv" << std::endl;
    }

    double totalMs = 0.0;
    for (int i = 0; i < opt.iters; ++i) {
        std::cout << "rank=" << opt.rank << " iter=" << i << " before Send/Recv" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (opt.rank == 0) {
            HCCLCHECK(HcclSend(devBuf, count, HCCL_DATA_TYPE_FP32, 1, comm, stream));
        } else {
            HCCLCHECK(HcclRecv(devBuf, count, HCCL_DATA_TYPE_FP32, 0, comm, stream));
        }
        ACLCHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "rank=" << opt.rank << " iter=" << i << " after Send/Recv" << std::endl;
    }

    const double avgMs = totalMs / opt.iters;
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const double gb = static_cast<double>(bytes) / 1e9;
    const double gibPerSec = gib / (avgMs / 1000.0);
    const double gbPerSec = gb / (avgMs / 1000.0);

    std::cout << "rank=" << opt.rank << " bytes=" << bytes << " avg_ms=" << avgMs
              << " throughput=" << gibPerSec << " GiB/s (" << gbPerSec << " GB/s)" << std::endl;

    if (hostBuf) {
        ACLCHECK(aclrtFreeHost(hostBuf));
    }
    ACLCHECK(aclrtFree(devBuf));
    HCCLCHECK(HcclCommDestroy(comm));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtResetDevice(opt.device));
    ACLCHECK(aclFinalize());
    return 0;
}
