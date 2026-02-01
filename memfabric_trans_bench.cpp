#include <acl/acl.h>
#include <smem.h>
#include <smem_shm.h>
#include <smem_trans.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_RET(expr, msg)                                                    \
    do {                                                                        \
        int32_t _ret = (expr);                                                  \
        if (_ret != 0) {                                                        \
            std::cerr << "[ERR] " << msg << " ret=" << _ret                     \
                      << " smem_err=" << (smem_get_last_err_msg() ?             \
                          smem_get_last_err_msg() : "null") << std::endl;       \
            return _ret;                                                        \
        }                                                                       \
    } while (0)

#define CHECK_ACL(expr, msg)                                                    \
    do {                                                                        \
        aclError _ret = (expr);                                                 \
        if (_ret != ACL_ERROR_NONE) {                                           \
            std::cerr << "[ACL ERR] " << msg << " ret=" << _ret << std::endl;   \
            return _ret;                                                        \
        }                                                                       \
    } while (0)

struct Options {
    int rank = 0;
    int world = 2;
    int device = 0;
    std::string storeUrl = "tcp://127.0.0.1:8570";
    std::string myId = "127.0.0.1:10001";
    std::string peerId = "127.0.0.1:10002";
    uint64_t bytes = (1ULL << 30);  // 1 GiB
    int warmup = 1;
    int iters = 5;
};

static void Usage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " --rank <0|1> --world 2 --device <id>"
              << " --store-url <tcp://ip:port> --my-id <ip:port> --peer-id <ip:port>"
              << " [--bytes <n>] [--warmup <n>] [--iters <n>]\n";
}

static bool ParseArgs(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--rank") {
            const char *v = need("--rank");
            if (!v) return false;
            opt.rank = std::atoi(v);
        } else if (key == "--world") {
            const char *v = need("--world");
            if (!v) return false;
            opt.world = std::atoi(v);
        } else if (key == "--device") {
            const char *v = need("--device");
            if (!v) return false;
            opt.device = std::atoi(v);
        } else if (key == "--store-url") {
            const char *v = need("--store-url");
            if (!v) return false;
            opt.storeUrl = v;
        } else if (key == "--my-id") {
            const char *v = need("--my-id");
            if (!v) return false;
            opt.myId = v;
        } else if (key == "--peer-id") {
            const char *v = need("--peer-id");
            if (!v) return false;
            opt.peerId = v;
        } else if (key == "--bytes") {
            const char *v = need("--bytes");
            if (!v) return false;
            opt.bytes = std::strtoull(v, nullptr, 10);
        } else if (key == "--warmup") {
            const char *v = need("--warmup");
            if (!v) return false;
            opt.warmup = std::atoi(v);
        } else if (key == "--iters") {
            const char *v = need("--iters");
            if (!v) return false;
            opt.iters = std::atoi(v);
        } else {
            std::cerr << "Unknown arg: " << key << std::endl;
            return false;
        }
    }
    return true;
}

static void FillIncreasing(float *buf, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        buf[i] = static_cast<float>(i + 1);
    }
}

static bool VerifySlice(const float *buf, uint64_t start, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(start + i + 1);
        if (buf[i] != expected) {
            std::cerr << "Verify failed at idx=" << (start + i)
                      << " got=" << buf[i] << " expected=" << expected << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseArgs(argc, argv, opt)) {
        Usage(argv[0]);
        return 1;
    }
    if (opt.world != 2 || (opt.rank != 0 && opt.rank != 1)) {
        std::cerr << "This sample expects --world 2 and --rank 0/1" << std::endl;
        return 1;
    }

    const uint64_t elemCount = opt.bytes / sizeof(float);
    if (elemCount % 4096 != 0) {
        std::cerr << "bytes must be divisible by 4096 * sizeof(float)" << std::endl;
        return 1;
    }
    const uint64_t cols = elemCount / 4096;
    std::cout << "shape=[4096," << cols << "] bytes=" << opt.bytes << std::endl;

    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(opt.device), "aclrtSetDevice");

    CHECK_RET(smem_init(0), "smem_init");
    if (opt.rank == 0) {
        CHECK_RET(smem_create_config_store(opt.storeUrl.c_str()), "smem_create_config_store");
    }

    smem_shm_config_t shm_cfg;
    CHECK_RET(smem_shm_config_init(&shm_cfg), "smem_shm_config_init");
    shm_cfg.startConfigStoreServer = false;
    CHECK_RET(smem_shm_init(opt.storeUrl.c_str(), opt.world, opt.rank, static_cast<uint16_t>(opt.device), &shm_cfg),
              "smem_shm_init");

    void *gva = nullptr;
    smem_shm_t shm = smem_shm_create(0, opt.world, opt.rank, 0, SMEMS_DATA_OP_MTE, 0, &gva);
    if (shm == nullptr) {
        std::cerr << "smem_shm_create failed: " << (smem_get_last_err_msg() ? smem_get_last_err_msg() : "null")
                  << std::endl;
        return 1;
    }
    CHECK_RET(smem_shm_control_barrier(shm), "shm barrier");

    smem_trans_config_t trans_cfg;
    CHECK_RET(smem_trans_config_init(&trans_cfg), "smem_trans_config_init");
    trans_cfg.deviceId = static_cast<uint32_t>(opt.device);
    trans_cfg.role = SMEM_TRANS_BOTH;
    trans_cfg.dataOpType = SMEMB_DATA_OP_DEVICE_RDMA;
    trans_cfg.startConfigServer = false;
    CHECK_RET(smem_trans_init(&trans_cfg), "smem_trans_init");

    smem_trans_t trans = smem_trans_create(opt.storeUrl.c_str(), opt.myId.c_str(), &trans_cfg);
    if (trans == nullptr) {
        std::cerr << "smem_trans_create failed: " << (smem_get_last_err_msg() ? smem_get_last_err_msg() : "null")
                  << std::endl;
        return 1;
    }

    void *dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, opt.bytes, ACL_MEM_MALLOC_HUGE_ONLY), "aclrtMalloc");
    std::cout << "rank=" << opt.rank << " dev_addr=" << dev << std::endl;

    void *gather_addrs[2] = {nullptr, nullptr};
    CHECK_RET(smem_shm_control_allgather(shm, reinterpret_cast<const char *>(&dev), sizeof(void *),
                                         reinterpret_cast<char *>(gather_addrs), sizeof(void *) * 2),
              "shm allgather");
    CHECK_RET(smem_shm_control_barrier(shm), "shm barrier after allgather");

    if (opt.rank == 1) {
        CHECK_RET(smem_trans_register_mem(trans, dev, opt.bytes, 0), "smem_trans_register_mem");
    }
    CHECK_RET(smem_shm_control_barrier(shm), "shm barrier after register");

    if (opt.rank == 0) {
        std::vector<float> host(elemCount);
        FillIncreasing(host.data(), elemCount);
        CHECK_ACL(aclrtMemcpy(dev, opt.bytes, host.data(), opt.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "H2D memcpy");

        for (int i = 0; i < opt.warmup; ++i) {
            CHECK_RET(smem_trans_write(trans, dev, opt.peerId.c_str(), gather_addrs[1], opt.bytes, 0),
                      "smem_trans_write warmup");
        }

        double total_ms = 0.0;
        for (int i = 0; i < opt.iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_RET(smem_trans_write(trans, dev, opt.peerId.c_str(), gather_addrs[1], opt.bytes, 0),
                      "smem_trans_write");
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double avg_ms = total_ms / opt.iters;
        double gib = static_cast<double>(opt.bytes) / (1024.0 * 1024.0 * 1024.0);
        double gb = static_cast<double>(opt.bytes) / 1e9;
        double gibps = gib / (avg_ms / 1000.0);
        double gbps = gb / (avg_ms / 1000.0);
        std::cout << "avg_ms=" << avg_ms << " throughput=" << gibps << " GiB/s (" << gbps << " GB/s)" << std::endl;
    }

    CHECK_RET(smem_shm_control_barrier(shm), "shm barrier before verify");

    if (opt.rank == 1) {
        const uint64_t k = 8;
        std::vector<float> head(k);
        std::vector<float> tail(k);
        CHECK_ACL(aclrtMemcpy(head.data(), k * sizeof(float), dev, k * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                  "D2H head");
        const uint64_t tail_offset = (elemCount - k) * sizeof(float);
        CHECK_ACL(aclrtMemcpy(tail.data(), k * sizeof(float),
                              reinterpret_cast<char *>(dev) + tail_offset, k * sizeof(float),
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "D2H tail");

        bool ok1 = VerifySlice(head.data(), 0, k);
        bool ok2 = VerifySlice(tail.data(), elemCount - k, k);
        std::cout << "verify_head=" << (ok1 ? "OK" : "FAIL")
                  << " verify_tail=" << (ok2 ? "OK" : "FAIL") << std::endl;
    }

    CHECK_RET(smem_shm_control_barrier(shm), "shm barrier before cleanup");

    if (opt.rank == 1) {
        (void)smem_trans_deregister_mem(trans, dev);
    }
    (void)aclrtFree(dev);
    (void)smem_trans_destroy(trans, 0);
    (void)smem_trans_uninit(0);
    (void)smem_shm_destroy(shm, 0);
    (void)smem_shm_uninit(0);
    (void)smem_uninit();
    (void)aclrtResetDevice(opt.device);
    (void)aclFinalize();
    return 0;
}
