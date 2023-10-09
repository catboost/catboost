#include "write_trie_backwards.h"

#include "comptrie_impl.h"
#include "leaf_skipper.h"

#include <util/generic/buffer.h>
#include <util/generic/vector.h>

namespace NCompactTrie {
    size_t WriteTrieBackwards(IOutputStream& os, TReverseNodeEnumerator& enumerator, bool verbose) {
        if (verbose) {
            Cerr << "Writing down the trie..." << Endl;
        }

        // Rewrite everything from the back, removing unused pieces.
        const size_t chunksize = 0x10000;
        TVector<char*> resultData;

        resultData.push_back(new char[chunksize]);
        char* chunkend = resultData.back() + chunksize;

        size_t resultLength = 0;
        size_t chunkLength = 0;

        size_t counter = 0;
        TBuffer bufferHolder;
        while (enumerator.Move()) {
            if (verbose)
                ShowProgress(++counter);

            size_t bufferLength = 64 + enumerator.GetLeafLength(); // never know how big leaf data can be
            bufferHolder.Clear();
            bufferHolder.Resize(bufferLength);
            char* buffer = bufferHolder.Data();

            size_t nodelength = enumerator.RecreateNode(buffer, resultLength);
            Y_ASSERT(nodelength <= bufferLength);

            resultLength += nodelength;

            if (chunkLength + nodelength <= chunksize) {
                chunkLength += nodelength;
                memcpy(chunkend - chunkLength, buffer, nodelength);
            } else { // allocate a new chunk
                memcpy(chunkend - chunksize, buffer + nodelength - (chunksize - chunkLength), chunksize - chunkLength);
                chunkLength = chunkLength + nodelength - chunksize;

                resultData.push_back(new char[chunksize]);
                chunkend = resultData.back() + chunksize;

                while (chunkLength > chunksize) { // allocate a new chunks
                    chunkLength -= chunksize;
                    memcpy(chunkend - chunksize, buffer + chunkLength, chunksize);

                    resultData.push_back(new char[chunksize]);
                    chunkend = resultData.back() + chunksize;
                }

                memcpy(chunkend - chunkLength, buffer, chunkLength);
            }
        }

        if (verbose)
            Cerr << counter << Endl;

        // Write the whole thing down
        while (!resultData.empty()) {
            char* chunk = resultData.back();
            os.Write(chunk + chunksize - chunkLength, chunkLength);
            chunkLength = chunksize;
            delete[] chunk;
            resultData.pop_back();
        }

        return resultLength;
    }

    size_t WriteTrieBackwardsNoAlloc(IOutputStream& os, TReverseNodeEnumerator& enumerator, TOpaqueTrie& trie, EMinimizeMode mode) {
        char* data = const_cast<char*>(trie.Data);
        char* end = data + trie.Length;
        char* pos = end;

        TVector<char> buf(64);
        while (enumerator.Move()) {
            size_t nodeLength = enumerator.RecreateNode(nullptr, end - pos);
            if (nodeLength > buf.size())
                buf.resize(nodeLength);

            size_t realLength = enumerator.RecreateNode(buf.data(), end - pos);
            Y_ASSERT(realLength == nodeLength);

            pos -= nodeLength;
            memcpy(pos, buf.data(), nodeLength);
        }

        switch (mode) {
            case MM_NOALLOC:
                os.Write(pos, end - pos);
                break;
            case MM_INPLACE:
                memmove(data, pos, end - pos);
                break;
            default:
                Y_ABORT_UNLESS(false);
        }

        return end - pos;
    }

}
