#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace jetson_fall {

// Small dependency-free JSON reader used for deployment configuration.  The
// runtime emits JSON through a dedicated serializer, so this parser only needs
// the standard object/array/string/number/bool/null subset.
struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Object, Array>;
    Storage value = nullptr;

    bool isObject() const { return std::holds_alternative<Object>(value); }
    bool isArray() const { return std::holds_alternative<Array>(value); }
    bool isString() const { return std::holds_alternative<std::string>(value); }
    bool isNumber() const { return std::holds_alternative<double>(value); }
    bool isBool() const { return std::holds_alternative<bool>(value); }
    const JsonValue* get(const std::string& key) const;
    std::string stringOr(const std::string& fallback) const;
    double numberOr(double fallback) const;
    bool boolOr(bool fallback) const;
};

JsonValue parseJson(const std::string& text);
std::string jsonEscape(const std::string& text);

}  // namespace jetson_fall
