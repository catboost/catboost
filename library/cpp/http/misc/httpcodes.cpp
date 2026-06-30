#include "httpcodes.h"

TStringBuf HttpCodeStrEx(int code) noexcept {
    switch (code) {
        case HTTP_CONTINUE:
            return TStringBuf("100 Continue");
        case HTTP_SWITCHING_PROTOCOLS:
            return TStringBuf("101 Switching protocols");
        case HTTP_PROCESSING:
            return TStringBuf("102 Processing");
        case HTTP_EARLY_HINTS:
            return TStringBuf ("103 Early Hints");

        case HTTP_OK:
            return TStringBuf("200 Ok");
        case HTTP_CREATED:
            return TStringBuf("201 Created");
        case HTTP_ACCEPTED:
            return TStringBuf("202 Accepted");
        case HTTP_NON_AUTHORITATIVE_INFORMATION:
            return TStringBuf("203 None authoritative information");
        case HTTP_NO_CONTENT:
            return TStringBuf("204 No content");
        case HTTP_RESET_CONTENT:
            return TStringBuf("205 Reset content");
        case HTTP_PARTIAL_CONTENT:
            return TStringBuf("206 Partial content");
        case HTTP_MULTI_STATUS:
            return TStringBuf("207 Multi status");
        case HTTP_ALREADY_REPORTED:
            return TStringBuf("208 Already reported");
        case HTTP_IM_USED:
            return TStringBuf("226 IM used");

        case HTTP_MULTIPLE_CHOICES:
            return TStringBuf("300 Multiple choices");
        case HTTP_MOVED_PERMANENTLY:
            return TStringBuf("301 Moved permanently");
        case HTTP_FOUND:
            return TStringBuf("302 Moved temporarily");
        case HTTP_SEE_OTHER:
            return TStringBuf("303 See other");
        case HTTP_NOT_MODIFIED:
            return TStringBuf("304 Not modified");
        case HTTP_USE_PROXY:
            return TStringBuf("305 Use proxy");
        case HTTP_TEMPORARY_REDIRECT:
            return TStringBuf("307 Temporarily redirect");
        case HTTP_PERMANENT_REDIRECT:
            return TStringBuf("308 Permanent redirect");

        case HTTP_BAD_REQUEST:
            return TStringBuf("400 Bad request");
        case HTTP_UNAUTHORIZED:
            return TStringBuf("401 Unauthorized");
        case HTTP_PAYMENT_REQUIRED:
            return TStringBuf("402 Payment required");
        case HTTP_FORBIDDEN:
            return TStringBuf("403 Forbidden");
        case HTTP_NOT_FOUND:
            return TStringBuf("404 Not found");
        case HTTP_METHOD_NOT_ALLOWED:
            return TStringBuf("405 Method not allowed");
        case HTTP_NOT_ACCEPTABLE:
            return TStringBuf("406 Not acceptable");
        case HTTP_PROXY_AUTHENTICATION_REQUIRED:
            return TStringBuf("407 Proxy Authentication required");
        case HTTP_REQUEST_TIME_OUT:
            return TStringBuf("408 Request time out");
        case HTTP_CONFLICT:
            return TStringBuf("409 Conflict");
        case HTTP_GONE:
            return TStringBuf("410 Gone");
        case HTTP_LENGTH_REQUIRED:
            return TStringBuf("411 Length required");
        case HTTP_PRECONDITION_FAILED:
            return TStringBuf("412 Precondition failed");
        case HTTP_REQUEST_ENTITY_TOO_LARGE:
            return TStringBuf("413 Request entity too large");
        case HTTP_REQUEST_URI_TOO_LARGE:
            return TStringBuf("414 Request uri too large");
        case HTTP_UNSUPPORTED_MEDIA_TYPE:
            return TStringBuf("415 Unsupported media type");
        case HTTP_REQUESTED_RANGE_NOT_SATISFIABLE:
            return TStringBuf("416 Requested Range Not Satisfiable");
        case HTTP_EXPECTATION_FAILED:
            return TStringBuf("417 Expectation Failed");
        case HTTP_I_AM_A_TEAPOT:
            return TStringBuf("418 I Am A Teapot");
        case HTTP_AUTHENTICATION_TIMEOUT:
            return TStringBuf("419 Authentication Timeout");
        case HTTP_MISDIRECTED_REQUEST:
            return TStringBuf("421 Misdirected Request");
        case HTTP_UNPROCESSABLE_ENTITY:
            return TStringBuf("422 Unprocessable Entity");
        case HTTP_LOCKED:
            return TStringBuf("423 Locked");
        case HTTP_FAILED_DEPENDENCY:
            return TStringBuf("424 Failed Dependency");
        case HTTP_UNORDERED_COLLECTION:
            return TStringBuf("425 Unordered Collection");
        case HTTP_UPGRADE_REQUIRED:
            return TStringBuf("426 Upgrade Required");
        case HTTP_PRECONDITION_REQUIRED:
            return TStringBuf("428 Precondition Required");
        case HTTP_TOO_MANY_REQUESTS:
            return TStringBuf("429 Too Many Requests");
        case HTTP_REQUEST_HEADER_FIELDS_TOO_LARGE:
            return TStringBuf("431 Request Header Fields Too Large");
        case HTTP_UNAVAILABLE_FOR_LEGAL_REASONS:
            return TStringBuf("451 Unavailable For Legal Reason");

        case HTTP_INTERNAL_SERVER_ERROR:
            return TStringBuf("500 Internal server error");
        case HTTP_NOT_IMPLEMENTED:
            return TStringBuf("501 Not implemented");
        case HTTP_BAD_GATEWAY:
            return TStringBuf("502 Bad gateway");
        case HTTP_SERVICE_UNAVAILABLE:
            return TStringBuf("503 Service unavailable");
        case HTTP_GATEWAY_TIME_OUT:
            return TStringBuf("504 Gateway time out");
        case HTTP_HTTP_VERSION_NOT_SUPPORTED:
            return TStringBuf("505 HTTP version not supported");
        case HTTP_VARIANT_ALSO_NEGOTIATES:
            return TStringBuf("506 Variant also negotiates");
        case HTTP_INSUFFICIENT_STORAGE:
            return TStringBuf("507 Insufficient storage");
        case HTTP_LOOP_DETECTED:
            return TStringBuf("508 Loop Detected");
        case HTTP_BANDWIDTH_LIMIT_EXCEEDED:
            return TStringBuf("509 Bandwidth Limit Exceeded");
        case HTTP_NOT_EXTENDED:
            return TStringBuf("510 Not Extended");
        case HTTP_NETWORK_AUTHENTICATION_REQUIRED:
            return TStringBuf("511 Network Authentication Required");
        case HTTP_UNASSIGNED_512:
            return TStringBuf("512 Unassigned");

        default:
            return TStringBuf("000 Unknown HTTP code");
    }
}
