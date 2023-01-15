#pragma once

enum ELogPriority {
    TLOG_EMERG = 0 /* "EMERG" */,
    TLOG_ALERT = 1 /* "ALERT" */,
    TLOG_CRIT = 2 /* "CRITICAL_INFO" */,
    TLOG_ERR = 3 /* "ERROR" */,
    TLOG_WARNING = 4 /* "WARNING" */,
    TLOG_NOTICE = 5 /* "NOTICE" */,
    TLOG_INFO = 6 /* "INFO" */,
    TLOG_DEBUG = 7 /* "DEBUG" */,
    TLOG_RESOURCES = 8 /* "RESOURCES" */
};
#define LOG_MAX_PRIORITY TLOG_RESOURCES
#define LOG_DEF_PRIORITY TLOG_INFO
