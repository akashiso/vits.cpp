#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <string>
#include <vector>

enum log_level_t
{
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
};

typedef void (*log_cb_t)(enum log_level_t level, const char* text, void* data);

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);
std::string get_full_path(const std::string& dir, const std::string& filename);

std::vector<std::string> get_files_from_dir(const std::string& dir);

int32_t get_num_physical_cores();

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::wstring utf8_to_utf16(const std::string& utf8_str);
std::string utf16_to_utf8(const std::wstring& utf16_str);
std::u32string unicode_value_to_utf32(int unicode_value);

std::string path_join(const std::string& p1, const std::string& p2);
std::vector<std::string> splitString(const std::string& str, char delimiter);
std::string trim(const std::string& s);

void log_printf(log_level_t level, const char* format, ...);
void set_log_callback(log_cb_t cb, void* data);
void get_log_callback(log_cb_t* cb, void** data);

void pretty_progress(int step, int steps, float time);

#define LOG_DEBUG(format, ...) log_printf(LOG_DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(LOG_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(LOG_WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(LOG_ERROR, format, ##__VA_ARGS__)
#endif  // __UTIL_H__
