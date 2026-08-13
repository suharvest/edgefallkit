#include "mini_json.h"

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace jetson_fall {
namespace {

class Parser {
public:
    explicit Parser(const std::string& input) : input_(input) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue result = parseValue();
        skipWhitespace();
        if (position_ != input_.size()) fail("trailing characters");
        return result;
    }

private:
    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(std::string("JSON parse error at ") +
                                 std::to_string(position_) + ": " + message);
    }

    void skipWhitespace() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skipWhitespace();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    JsonValue parseValue() {
        skipWhitespace();
        if (position_ >= input_.size()) fail("unexpected end");
        switch (input_[position_]) {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return JsonValue{parseString()};
            case 't': parseLiteral("true"); return JsonValue{true};
            case 'f': parseLiteral("false"); return JsonValue{false};
            case 'n': parseLiteral("null"); return JsonValue{nullptr};
            default:
                if (input_[position_] == '-' || std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                    return JsonValue{parseNumber()};
                }
                fail("unexpected token");
        }
    }

    JsonValue parseObject() {
        if (!consume('{')) fail("expected '{'");
        JsonValue::Object object;
        skipWhitespace();
        if (consume('}')) return JsonValue{std::move(object)};
        while (true) {
            skipWhitespace();
            if (position_ >= input_.size() || input_[position_] != '"') fail("object key must be string");
            const std::string key = parseString();
            if (!consume(':')) fail("expected ':'");
            object.emplace(key, parseValue());
            if (consume('}')) break;
            if (!consume(',')) fail("expected ','");
        }
        return JsonValue{std::move(object)};
    }

    JsonValue parseArray() {
        if (!consume('[')) fail("expected '['");
        JsonValue::Array array;
        skipWhitespace();
        if (consume(']')) return JsonValue{std::move(array)};
        while (true) {
            array.push_back(parseValue());
            if (consume(']')) break;
            if (!consume(',')) fail("expected ','");
        }
        return JsonValue{std::move(array)};
    }

    void parseLiteral(const char* literal) {
        while (*literal != '\0') {
            if (position_ >= input_.size() || input_[position_] != *literal) fail("invalid literal");
            ++position_;
            ++literal;
        }
    }

    std::string parseString() {
        if (position_ >= input_.size() || input_[position_] != '"') fail("expected string");
        ++position_;
        std::string result;
        while (position_ < input_.size()) {
            const char current = input_[position_++];
            if (current == '"') return result;
            if (current != '\\') {
                result.push_back(current);
                continue;
            }
            if (position_ >= input_.size()) fail("unfinished escape");
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    // Configuration values are ASCII URLs/paths/topics in
                    // practice. Decode a BMP escape into UTF-8 for completeness.
                    if (position_ + 4 > input_.size()) fail("short unicode escape");
                    unsigned codepoint = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char c = input_[position_++];
                        codepoint <<= 4;
                        if (c >= '0' && c <= '9') codepoint += static_cast<unsigned>(c - '0');
                        else if (c >= 'a' && c <= 'f') codepoint += static_cast<unsigned>(c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') codepoint += static_cast<unsigned>(c - 'A' + 10);
                        else fail("invalid unicode escape");
                    }
                    if (codepoint < 0x80) result.push_back(static_cast<char>(codepoint));
                    else if (codepoint < 0x800) {
                        result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
                        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
                    } else {
                        result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
                        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
                        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
                    }
                    break;
                }
                default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    double parseNumber() {
        const std::size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) fail("invalid number");
        if (input_[position_] == '0') ++position_;
        else {
            if (!std::isdigit(static_cast<unsigned char>(input_[position_]))) fail("invalid number");
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            if (position_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[position_]))) fail("invalid fraction");
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            if (position_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[position_]))) fail("invalid exponent");
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        char* end = nullptr;
        const double result = std::strtod(input_.c_str() + start, &end);
        if (end != input_.c_str() + position_) fail("invalid number");
        return result;
    }

    const std::string& input_;
    std::size_t position_ = 0;
};

}  // namespace

const JsonValue* JsonValue::get(const std::string& key) const {
    if (!isObject()) return nullptr;
    const auto& object = std::get<Object>(value);
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

std::string JsonValue::stringOr(const std::string& fallback) const {
    return isString() ? std::get<std::string>(value) : fallback;
}

double JsonValue::numberOr(double fallback) const {
    return isNumber() ? std::get<double>(value) : fallback;
}

bool JsonValue::boolOr(bool fallback) const {
    return isBool() ? std::get<bool>(value) : fallback;
}

JsonValue parseJson(const std::string& text) { return Parser(text).parse(); }

std::string jsonEscape(const std::string& text) {
    std::ostringstream output;
    for (const unsigned char c : text) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                           << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

}  // namespace jetson_fall
