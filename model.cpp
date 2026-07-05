#include <stdarg.h>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "util.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#define ST_HEADER_SIZE_LEN 8

uint64_t read_u64(uint8_t* buffer) {
    // little endian
    uint64_t value = 0;
    value |= static_cast<int64_t>(buffer[7]) << 56;
    value |= static_cast<int64_t>(buffer[6]) << 48;
    value |= static_cast<int64_t>(buffer[5]) << 40;
    value |= static_cast<int64_t>(buffer[4]) << 32;
    value |= static_cast<int64_t>(buffer[3]) << 24;
    value |= static_cast<int64_t>(buffer[2]) << 16;
    value |= static_cast<int64_t>(buffer[1]) << 8;
    value |= static_cast<int64_t>(buffer[0]);
    return value;
}

int32_t read_int(uint8_t* buffer) {
    // little endian
    int value = 0;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

uint16_t read_short(uint8_t* buffer) {
    // little endian
    uint16_t value = 0;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

float bf16_to_f32(uint16_t bfloat16) {
    uint32_t val_bits = (static_cast<uint32_t>(bfloat16) << 16);
    return *reinterpret_cast<float*>(&val_bits);
}

uint16_t f8_e4m3_to_f16(uint8_t f8) {
    // do we need to support uz?

    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return ggml_fp32_to_fp16(-NAN);
    } else if (f8 == 0x7f) {
        return ggml_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;

            // yes, 2 times
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }

            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }

    return ggml_fp32_to_fp16(*reinterpret_cast<const float*>(&result));
}

uint16_t f8_e5m2_to_f16(uint8_t fp8) {
    uint8_t sign     = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 2) & 0x1F;
    uint8_t mantissa = fp8 & 0x3;

    uint16_t fp16_sign = sign << 15;
    uint16_t fp16_exponent;
    uint16_t fp16_mantissa;

    if (exponent == 0 && mantissa == 0) {  // zero
        return fp16_sign;
    }

    if (exponent == 0x1F) {  // NAN and INF
        fp16_exponent = 0x1F;
        fp16_mantissa = mantissa ? (mantissa << 8) : 0;
        return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
    }

    if (exponent == 0) {  // subnormal numbers
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
        return fp16_sign | fp16_mantissa;
    }

    // normal numbers
    int16_t true_exponent = (int16_t)exponent - 15 + 15;
    if (true_exponent <= 0) {
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
    } else if (true_exponent >= 0x1F) {
        fp16_exponent = 0x1F;
        fp16_mantissa = 0;
    } else {
        fp16_exponent = (uint16_t)true_exponent;
        fp16_mantissa = mantissa << 8;
    }

    return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
}

void bf16_to_f32_vec(uint16_t* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = bf16_to_f32(src[i]);
    }
}

void f8_e4m3_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e4m3_to_f16(src[i]);
    }
}
void f8_e5m2_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e5m2_to_f16(src[i]);
    }
}

void convert_tensor(void* src,
                    ggml_type src_type,
                    void* dst,
                    ggml_type dst_type,
                    int nrows,
                    int n_per_row) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_get_type_traits(src_type);
            if (qtype->to_float == NULL) {
                throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                                ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                            ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype->to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}

/*================================================= ModelLoader ==================================================*/

bool is_zip_file(const std::string& file_path) {
    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        return false;
    }
    zip_close(zip);
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_zip_file(file_path)) {
        LOG_INFO("load %s using checkpoint format", file_path.c_str());
        return init_from_ckpt_file(file_path, prefix);
    } else {
        LOG_WARN("unknown format %s", file_path.c_str());
        return false;
    }
}

/*================================================= GGUFModelLoader ==================================================*/

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    gguf_context* ctx_gguf_ = NULL;
    ggml_context* ctx_meta_ = NULL;
    ctx_gguf_               = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf_);

    size_t total_size  = 0;
    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name          = gguf_get_tensor_name(ctx_gguf_, i);
        struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset             = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        // LOG_DEBUG("%s", name.c_str());

        TensorStorage tensor_storage(prefix + name, dummy->type, dummy->ne, ggml_n_dims(dummy), file_index, offset);

        GGML_ASSERT(ggml_nbytes(dummy) == tensor_storage.nbytes());

        tensor_storages.push_back(tensor_storage);
        tensor_storages_types[tensor_storage.name] = tensor_storage.type;
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

/*================================================= CkptModelLoader ==================================================*/

// $ python -m pickletools sd-v1-4/archive/data.pkl | head -n 100
//     0: \x80 PROTO      2
//     2: }    EMPTY_DICT
//     3: q    BINPUT     0
//     5: (    MARK
//     6: X        BINUNICODE 'epoch'
//    16: q        BINPUT     1
//    18: K        BININT1    6
//    20: X        BINUNICODE 'global_step'
//    36: q        BINPUT     2
//    38: J        BININT     470000
//    43: X        BINUNICODE 'pytorch-lightning_version'
//    73: q        BINPUT     3
//    75: X        BINUNICODE '1.4.2'
//    85: q        BINPUT     4
//    87: X        BINUNICODE 'state_dict'
//   102: q        BINPUT     5
//   104: }        EMPTY_DICT
//   105: q        BINPUT     6
//   107: (        MARK
//   108: X            BINUNICODE 'betas'
//   118: q            BINPUT     7
//   120: c            GLOBAL     'torch._utils _rebuild_tensor_v2'
//   153: q            BINPUT     8
//   155: (            MARK
//   156: (                MARK
//   157: X                    BINUNICODE 'storage'
//   169: q                    BINPUT     9
//   171: c                    GLOBAL     'torch FloatStorage'
//   191: q                    BINPUT     10
//   193: X                    BINUNICODE '0'
//   199: q                    BINPUT     11
//   201: X                    BINUNICODE 'cpu'
//   209: q                    BINPUT     12
//   211: M                    BININT2    1000
//   214: t                    TUPLE      (MARK at 156)
//   215: q                BINPUT     13
//   217: Q                BINPERSID
//   218: K                BININT1    0
//   220: M                BININT2    1000
//  ...............................
//  3201: q            BINPUT     250
//  3203: R            REDUCE
//  3204: q            BINPUT     251
//  3206: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.weight'
//  3264: q            BINPUT     252
//  3266: h            BINGET     8
//  3268: (            MARK
//  3269: (                MARK
//  3270: h                    BINGET     9
//  3272: h                    BINGET     10
//  3274: X                    BINUNICODE '30'
//  3281: q                    BINPUT     253
//  3283: h                    BINGET     12
//  3285: J                    BININT     102400
//  3290: t                    TUPLE      (MARK at 3269)
//  3291: q                BINPUT     254
//  3293: Q                BINPERSID
//  3294: K                BININT1    0
//  3296: (                MARK
//  3297: M                    BININT2    320
//  3300: M                    BININT2    320
//  3303: K                    BININT1    1
//  3305: K                    BININT1    1
//  3307: t                    TUPLE      (MARK at 3296)
//  3308: q                BINPUT     255
//  3310: (                MARK
//  3311: M                    BININT2    320
//  3314: K                    BININT1    1
//  3316: K                    BININT1    1
//  3318: K                    BININT1    1
//  3320: t                    TUPLE      (MARK at 3310)
//  3321: r                LONG_BINPUT 256
//  3326: \x89             NEWFALSE
//  3327: h                BINGET     16
//  3329: )                EMPTY_TUPLE
//  3330: R                REDUCE
//  3331: r                LONG_BINPUT 257
//  3336: t                TUPLE      (MARK at 3268)
//  3337: r            LONG_BINPUT 258
//  3342: R            REDUCE
//  3343: r            LONG_BINPUT 259
//  3348: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.bias'
//  3404: r            LONG_BINPUT 260
//  3409: h            BINGET     8
//  3411: (            MARK
//  3412: (                MARK
//  3413: h                    BINGET     9
//  3415: h                    BINGET     10
//  3417: X                    BINUNICODE '31'

struct PickleTensorReader {
    enum ReadPhase {
        READ_NAME,
        READ_DATA,
        CHECK_SIZE,
        READ_DIMENS
    };
    ReadPhase phase   = READ_NAME;
    size_t entry_size = 0;
    int32_t nelements = 0;

    TensorStorage tensor_storage;

    static ggml_type global_type;  // all pickle_tensors data type
    static bool read_global_type;

    bool read_int_value(uint32_t value) {
        if (phase == CHECK_SIZE) {
            if (entry_size == value * ggml_type_size(tensor_storage.type)) {
                nelements = value;
                phase     = READ_DIMENS;
                return true;
            } else {
                phase = READ_NAME;
            }
        } else if (phase == READ_DIMENS) {
            if (tensor_storage.n_dims + 1 > 5) {  // too many dimens
                phase                 = READ_NAME;
                tensor_storage.n_dims = 0;
            }
            if (nelements % value == 0) {
                tensor_storage.ne[tensor_storage.n_dims] = value;
                tensor_storage.n_dims++;
            }
        }
        return false;
    }

    void read_global(const std::string& str) {
        if (str == "FloatStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F32;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F32;
        } else if (str == "HalfStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F16;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F16;
        }
    }

    void read_string(const std::string& str, struct zip_t* zip, std::string dir) {
        if (str == "storage") {
            read_global_type = true;
        } else if (str != "state_dict") {
            if (phase == READ_DATA) {
                std::string entry_name = dir + "data/" + std::string(str);

                size_t i, n = zip_entries_total(zip);
                for (i = 0; i < n; ++i) {
                    zip_entry_openbyindex(zip, i);
                    {
                        std::string name = zip_entry_name(zip);
                        if (name == entry_name) {
                            tensor_storage.index_in_zip = (int)i;
                            entry_size                  = zip_entry_size(zip);
                            zip_entry_close(zip);
                            break;
                        }
                    }
                    zip_entry_close(zip);
                }

                phase = entry_size > 0 ? CHECK_SIZE : READ_NAME;
            }
            if (!read_global_type && phase == READ_NAME) {
                tensor_storage.name = str;
                phase               = READ_DATA;
                tensor_storage.type = global_type;
            }
        }
    }
};

ggml_type PickleTensorReader::global_type = GGML_TYPE_F32;  // all pickle_tensors data type
bool PickleTensorReader::read_global_type = false;

int find_char(uint8_t* buffer, int len, char c) {
    for (int pos = 0; pos < len; pos++) {
        if (buffer[pos] == c) {
            return pos;
        }
    }
    return -1;
}

#define MAX_STRING_BUFFER 512

bool ModelLoader::parse_data_pkl(uint8_t* buffer,
                                 size_t buffer_size,
                                 zip_t* zip,
                                 std::string dir,
                                 size_t file_index,
                                 const std::string prefix) {
    uint8_t* buffer_end = buffer + buffer_size;
    if (buffer[0] == 0x80) {  // proto
        if (buffer[1] != 2) {
            LOG_ERROR("Unsupported protocol\n");
            return false;
        }
        buffer += 2;  // 0x80 and version
        char string_buffer[MAX_STRING_BUFFER];
        bool finish = false;
        PickleTensorReader reader;
        // read pickle binary file
        while (!finish && buffer < buffer_end) {
            uint8_t opcode = *buffer;
            buffer++;
            // https://github.com/python/cpython/blob/3.7/Lib/pickletools.py#L1048
            // https://github.com/python/cpython/blob/main/Lib/pickle.py#L105
            switch (opcode) {
                case '}':  // EMPTY_DICT     = b'}'   # push empty dict
                    break;
                case ']':  // EMPTY_LIST     = b']'   # push empty list
                    break;
                // skip unused sections
                case 'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
                case 'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
                case 'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
                    buffer++;
                    break;
                case 'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
                    buffer += 4;
                    break;
                case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                    buffer += 8;
                    break;
                case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
                    break;
                case '(':  // MARK           = b'('   # push special markobject on stack
                    break;
                case 'K':  // BININT1        = b'K'   # push 1-byte unsigned int
                {
                    uint8_t value = *buffer;
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer++;
                } break;
                case 'M':  // BININT2        = b'M'   # push 2-byte unsigned int
                {
                    uint16_t value = read_short(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer += 2;
                } break;
                case 'J':  // BININT         = b'J'   # push four-byte signed int
                {
                    const int32_t value = read_int(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;  // skip tuple after read num_elements
                    }
                    buffer += 4;
                } break;
                case 'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
                {
                    const int32_t len = read_int(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    if (len > MAX_STRING_BUFFER) {
                        LOG_WARN("tensor name very large");
                    }
                    memcpy(string_buffer, buffer, len < MAX_STRING_BUFFER ? len : (MAX_STRING_BUFFER - 1));
                    buffer += len;
                    reader.read_string(string_buffer, zip, dir);
                } break;
                case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
                {
                    const int8_t len = *buffer;
                    buffer++;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    // printf("String: '%s'\n", string_buffer);
                } break;
                case 'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
                {
                    int len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    buffer += len + 1;
                    len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len + 1;
                    reader.read_global(string_buffer);
                } break;
                case 0x87:  // TUPLE3         = b'\x87'  # build 3-tuple from three topmost stack items
                case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
                case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
                case 't':   // TUPLE          = b't'   # build tuple from topmost stack items
                    if (reader.phase == PickleTensorReader::READ_DIMENS) {
                        reader.tensor_storage.reverse_ne();
                        reader.tensor_storage.file_index = file_index;
                        // if(strcmp(prefix.c_str(), "scarlett") == 0)
                        // printf(" ZIP got tensor %s \n ", reader.tensor_storage.name.c_str());
                        reader.tensor_storage.name = prefix + reader.tensor_storage.name;
                        tensor_storages.push_back(reader.tensor_storage);
                        tensor_storages_types[reader.tensor_storage.name] = reader.tensor_storage.type;

                        // LOG_DEBUG("%s", reader.tensor_storage.name.c_str());
                        // reset
                        reader = PickleTensorReader();
                    }
                    break;
                case '.':  // STOP           = b'.'   # every pickle ends with STOP
                    finish = true;
                    break;
                default:
                    break;
            }
        }
    }
    return true;
}

bool ModelLoader::init_from_ckpt_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }
    int n = (int)zip_entries_total(zip);
    for (int i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        {
            std::string name = zip_entry_name(zip);
            size_t pos       = name.find("data.pkl");
            if (pos != std::string::npos) {
                std::string dir = name.substr(0, pos);
                printf("ZIP %d, name = %s, dir = %s \n", i, name.c_str(), dir.c_str());
                void* pkl_data = NULL;
                size_t pkl_size;
                zip_entry_read(zip, &pkl_data, &pkl_size);

                // LOG_DEBUG("%lld", pkl_size);

                parse_data_pkl((uint8_t*)pkl_data, pkl_size, zip, dir, file_index, prefix);

                free(pkl_data);
            }
        }
        zip_entry_close(zip);
    }
    zip_close(zip);
    return true;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string prefix) {
    for (auto& pair : tensor_storages_types) {
        if (prefix.size() < 1 || pair.first.substr(0, prefix.size()) == prefix) {
            for (auto& tensor_storage : tensor_storages) {
                if (tensor_storage.name == pair.first) {
                    if (tensor_should_be_converted(tensor_storage, wtype)) {
                        pair.second = wtype;
                    }
                    break;
                }
            }
        }
    }
}

std::vector<TensorStorage> remove_duplicates(const std::vector<TensorStorage>& vec) {
    std::vector<TensorStorage> res;
    std::unordered_map<std::string, size_t> name_to_index_map;

    for (size_t i = 0; i < vec.size(); ++i) {
        const std::string& current_name = vec[i].name;
        auto it                         = name_to_index_map.find(current_name);

        if (it != name_to_index_map.end()) {
            res[it->second] = vec[i];
        } else {
            name_to_index_map[current_name] = i;
            res.push_back(vec[i]);
        }
    }

    // vec.resize(name_to_index_map.size());

    return res;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, ggml_backend_t backend) {
    std::vector<TensorStorage> processed_tensor_storages = tensor_storages;
    std::vector<TensorStorage> dedup = remove_duplicates(processed_tensor_storages);
    processed_tensor_storages        = dedup;

    bool success = true;
    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];
        LOG_DEBUG("loading tensors from %s", file_path.c_str());

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        bool is_zip = false;
        for (auto& tensor_storage : tensor_storages) {
            if (tensor_storage.file_index != file_index) {
                continue;
            }
            if (tensor_storage.index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        struct zip_t* zip = NULL;
        if (is_zip) {
            zip = zip_open(file_path.c_str(), 0, 'r');
            if (zip == NULL) {
                LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                return false;
            }
        }

        std::vector<uint8_t> read_buffer;
        std::vector<uint8_t> convert_buffer;

        auto read_data = [&](const TensorStorage& tensor_storage, char* buf, size_t n) {
            if (zip != NULL) {
                zip_entry_openbyindex(zip, tensor_storage.index_in_zip);
                size_t entry_size = zip_entry_size(zip);
                if (entry_size != n) {
                    read_buffer.reserve(entry_size);
                    zip_entry_noallocread(zip, (void*)read_buffer.data(), entry_size);
                    memcpy((void*)buf, (void*)(read_buffer.data() + tensor_storage.offset), n);
                } else {
                    zip_entry_noallocread(zip, (void*)buf, n);
                }
                zip_entry_close(zip);
            } else {
                file.seekg(tensor_storage.offset);
                file.read(buf, n);
                if (!file) {
                    LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                    return false;
                }
            }
            return true;
        };
        int tensor_count = 0;
        int64_t t1       = ggml_time_ms();
        for (auto& tensor_storage : processed_tensor_storages) {
            if (tensor_storage.file_index != file_index) {
                ++tensor_count;
                continue;
            }
            ggml_tensor* dst_tensor = NULL;

            success = on_new_tensor_cb(tensor_storage, &dst_tensor);
            if (!success) {
                LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                break;
            }

            if (dst_tensor == NULL) {
                ++tensor_count;
                continue;
            }

            size_t nbytes_to_read = tensor_storage.nbytes_to_read();

            if (dst_tensor->buffer == NULL || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                // for the CPU and Metal backend, we can copy directly into the tensor
                if (tensor_storage.type == dst_tensor->type) {
                    GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                    read_data(tensor_storage, (char*)dst_tensor->data, nbytes_to_read);

                    if (tensor_storage.is_bf16) {
                        // inplace op
                        bf16_to_f32_vec((uint16_t*)dst_tensor->data, (float*)dst_tensor->data, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e4m3) {
                        // inplace op
                        f8_e4m3_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        // inplace op
                        f8_e5m2_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                    }
                } else {
                    read_buffer.reserve(tensor_storage.nbytes());
                    read_data(tensor_storage, (char*)read_buffer.data(), nbytes_to_read);

                    if (tensor_storage.is_bf16) {
                        // inplace op
                        bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e4m3) {
                        // inplace op
                        f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        // inplace op
                        f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                    }

                    convert_tensor((void*)read_buffer.data(), tensor_storage.type, dst_tensor->data,
                                   dst_tensor->type, (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                }
            } else {
                read_buffer.reserve(tensor_storage.nbytes());
                read_data(tensor_storage, (char*)read_buffer.data(), nbytes_to_read);

                if (tensor_storage.is_bf16) {
                    // inplace op
                    bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                } else if (tensor_storage.is_f8_e4m3) {
                    // inplace op
                    f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                } else if (tensor_storage.is_f8_e5m2) {
                    // inplace op
                    f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                }

                if (tensor_storage.type == dst_tensor->type) {
                    // copy to device memory
                    ggml_backend_tensor_set(dst_tensor, read_buffer.data(), 0, ggml_nbytes(dst_tensor));
                } else {
                    // convert first, then copy to device memory
                    convert_buffer.reserve(ggml_nbytes(dst_tensor));
                    convert_tensor((void*)read_buffer.data(), tensor_storage.type,
                                   (void*)convert_buffer.data(), dst_tensor->type,
                                   (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                    ggml_backend_tensor_set(dst_tensor, convert_buffer.data(), 0, ggml_nbytes(dst_tensor));
                }
            }
            int64_t t2 = ggml_time_ms();
            pretty_progress(++tensor_count, processed_tensor_storages.size(), (t2 - t1) / 1000.0f);
            t1 = t2;
        }

        if (zip != NULL) {
            zip_close(zip);
        }

        if (!success) {
            break;
        }
    }
    return success;
}

bool ModelLoader::load_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                               ggml_backend_t backend,
                               std::set<std::string> ignore_tensors) {
    std::set<std::string> tensor_names_in_file;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
        tensor_names_in_file.insert(name);

        struct ggml_tensor* real;
        if (tensors.find(name) != tensors.end()) {
            real = tensors[name];
        } else {
            for (auto& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        if (
            real->ne[0] != tensor_storage.ne[0] ||
            real->ne[1] != tensor_storage.ne[1] ||
            real->ne[2] != tensor_storage.ne[2] ||
            real->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "tensor '%s' has wrong shape in model file: "
                "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
            return false;
        }

        *dst_tensor = real;

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, backend);
    if (!success) {
        LOG_ERROR("load tensors from file failed");
        return false;
    }

    bool some_tensor_not_init = false;

    for (auto pair : tensors) {
        if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
            some_tensor_not_init = true;
        }
    }

    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
    const std::string& name = tensor_storage.name;
    if (type != GGML_TYPE_COUNT) {
        if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
            // Pass, do not convert
        } else if (ends_with(name, ".bias")) {

        } else {
            return true;
        }
    }
    return false;
}

bool ModelLoader::save_to_gguf_file(const std::string& file_path, ggml_type type) {
    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += tensor_storages.size() * ggml_tensor_overhead();
    mem_size += get_params_mem_size(backend, type);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, NULL, false});

    gguf_context* gguf_ctx = gguf_init_empty();

    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;

        ggml_type tensor_type = tensor_storage.type;
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_type = type;
        }

        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == NULL) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        // LOG_DEBUG("%s %d %s %d[%d %d %d %d] %d[%d %d %d %d]", name.c_str(),
        // ggml_nbytes(tensor), ggml_type_name(tensor_type),
        // tensor_storage.n_dims,
        // tensor_storage.ne[0], tensor_storage.ne[1], tensor_storage.ne[2], tensor_storage.ne[3],
        // tensor->n_dims, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

        *dst_tensor = tensor;

        gguf_add_tensor(gguf_ctx, tensor);

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, backend);
    ggml_backend_free(backend);
    LOG_INFO("load tensors done");
    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    if (success) {
        gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    }
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
    return success;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) {
    size_t alignment = 128;
    if (backend != NULL) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    std::vector<TensorStorage> processed_tensor_storages = tensor_storages;

    for (auto& tensor_storage : processed_tensor_storages) {
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}

bool convert(const char* input_path, const char* output_path, ggml_type output_type) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    bool success = model_loader.save_to_gguf_file(output_path, (ggml_type)output_type);
    return success;
}
