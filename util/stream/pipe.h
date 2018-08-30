#pragma once

#include "input.h"
#include "output.h"

#include <util/system/pipe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>

/**
 * @addtogroup Streams_Pipes
 * @{
 */

/**
 * Base class for starting a process and communicating with it via pipes.
 */
class TPipeBase {
protected:
    /**
     * Starts a new process and opens a pipe.
     *
     * @param command                   Command line to start a process with.
     * @param mode                      Data transfer mode for the pipe. Use
     *                                  "r" for reading and "w" for writing.
     */
    TPipeBase(const TString& command, const char* mode);
    virtual ~TPipeBase();

protected:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * Input stream that binds to a standard output stream of a newly started process.
 *
 * Note that if the process ends with non-zero exit status, `Read` function will
 * throw an exception.
 */
class TPipeInput: protected TPipeBase, public IInputStream {
public:
    /**
     * Starts a new process and opens a pipe.
     *
     * @param command                   Command line to start a process with.
     */
    TPipeInput(const TString& command);

private:
    size_t DoRead(void* buf, size_t len) override;
};

/**
 * Output stream that binds to a standard input stream of a newly started process.
 *
 * Note that if the process ends with non-zero exit status, `Close` function will
 * throw an exception.
 */
class TPipeOutput: protected TPipeBase, public IOutputStream {
public:
    /**
     * Starts a new process and opens a pipe.
     *
     * @param command                   Command line to start a process with.
     */
    TPipeOutput(const TString& command);

    /**
     * Waits for the process to terminate and throws an exception if it ended
     * with a non-zero exit status.
     */
    void Close();

private:
    void DoWrite(const void* buf, size_t len) override;
};

class TPipedBase {
protected:
    TPipedBase(PIPEHANDLE fd);
    virtual ~TPipedBase();

protected:
    TPipeHandle Handle_;
};

/**
 * Input stream that binds to a standard output stream of an existing process.
 */
class TPipedInput: public TPipedBase, public IInputStream {
public:
    TPipedInput(PIPEHANDLE fd);
    ~TPipedInput() override;

private:
    size_t DoRead(void* buf, size_t len) override;
};

/**
 * Output stream that binds to a standard input stream of an existing process.
 */
class TPipedOutput: public TPipedBase, public IOutputStream {
public:
    TPipedOutput(PIPEHANDLE fd);
    ~TPipedOutput() override;

private:
    void DoWrite(const void* buf, size_t len) override;
};

/** @} */
