#include "dirut.h"
#include "iterator.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/fs.h>
#include <util/system/file.h>
#include <util/generic/hash.h>
#include <util/generic/algorithm.h>
#include <util/random/mersenne.h>

static TString JoinWithNewline(const TVector<TString>& strings) {
    TStringStream ss;
    for (const auto& string : strings) {
        ss << string << "\n";
    }
    return ss.Str();
}

class TDirIteratorTest: public TTestBase {
    UNIT_TEST_SUITE(TDirIteratorTest);
    UNIT_TEST(TestIt)
    UNIT_TEST(TestError)
    UNIT_TEST(TestLocal)
    UNIT_TEST(TestSkip)
    UNIT_TEST(TestSort)
    UNIT_TEST_SUITE_END();

private:
    class TDirHier {
    public:
        struct TPath {
            TString Path;
            int Type;
        };

        inline void AddFile(const TString& path) {
            Add(path, 0);
        }

        inline void AddDir(const TString& path) {
            Add(path, 1);
        }

        inline void Add(const TString& path, int type) {
            const TPath p = {
                path, type};

            Add(p);
        }

        inline void Add(const TPath& path) {
            switch (path.Type) {
                case 0:
                    TFile(path.Path, CreateAlways | RdWr);
                    break;

                case 1:
                    MakeDirIfNotExist(path.Path.data());
                    break;

                case 2:
                    ythrow yexception() << "unknown path type";
            }

            Paths_.push_back(path);
            Srch_[path.Path] = path;
        }

        inline int Type(const TString& path) {
            THashMap<TString, TPath>::const_iterator it = Srch_.find(path);

            UNIT_ASSERT(it != Srch_.end());

            return it->second.Type;
        }

        inline bool Have(const TString& path, int type) {
            return Type(path) == type;
        }

        inline ~TDirHier() {
            for (size_t i = 0; i < Paths_.size(); ++i) {
                NFs::Remove(Paths_[Paths_.size() - i - 1].Path);
            }
        }

    private:
        TVector<TPath> Paths_;
        THashMap<TString, TPath> Srch_;
    };

    inline void TestLocal() {
        TString dirname("." LOCSLASH_S);
        TDirIterator d(dirname, FTS_NOCHDIR);
        for (auto it = d.begin(); it != d.end(); ++it) {
        }
    }

    inline void TestIt() {
        TDirHier hier;

        const TString dir = "tmpdir";
        const TDirHier::TPath path = {dir, 1};

        hier.Add(path);

        for (size_t i = 0; i < 10; ++i) {
            const TString dir1 = dir + LOCSLASH_C + ToString(i);
            const TDirHier::TPath path1 = {dir1, 1};

            hier.Add(path1);

            for (size_t j = 0; j < 10; ++j) {
                const TString subdir2 = ToString(j);
                const TString dir2 = dir1 + LOCSLASH_C + subdir2;
                const TDirHier::TPath path2 = {dir2, 1};

                hier.Add(path2);

                for (size_t k = 0; k < 3; ++k) {
                    const TString file = dir2 + LOCSLASH_C + "file" + ToString(k);
                    const TDirHier::TPath fpath = {file, 0};

                    hier.Add(fpath);
                }
            }
        }

        TDirIterator d(dir);

        for (auto it = d.begin(); it != d.end(); ++it) {
            UNIT_ASSERT(hier.Have(it->fts_path, it->fts_info != FTS_F));
        }
    }

    inline void TestSkip() {
        TDirHier hier;

        const TString dir = "tmpdir";
        const TDirHier::TPath path = {dir, 1};

        hier.Add(path);
        hier.AddDir(dir + LOCSLASH_C + "dir1");
        hier.AddDir(dir + LOCSLASH_C + "dir1" + LOCSLASH_C + "dir2");
        //
        // Without skip
        //
        {
            TDirIterator di(dir);

            UNIT_ASSERT(di.Next());
            UNIT_ASSERT_EQUAL(TStringBuf(di.Next()->fts_name), "dir1");
            UNIT_ASSERT_EQUAL(TStringBuf(di.Next()->fts_name), "dir2");
            UNIT_ASSERT_EQUAL(TStringBuf(di.Next()->fts_name), "dir2");
            UNIT_ASSERT_EQUAL(TStringBuf(di.Next()->fts_name), "dir1");
            UNIT_ASSERT(di.Next());
            UNIT_ASSERT_EQUAL(di.Next(), nullptr);
        }
        //
        // With skip
        //
        {
            TDirIterator di(dir);

            UNIT_ASSERT(di.Next());
            auto ent = di.Next();
            UNIT_ASSERT_EQUAL(TStringBuf(ent->fts_name), "dir1");
            di.Skip(ent);
            UNIT_ASSERT_EQUAL(TStringBuf(di.Next()->fts_name), "dir1");
            UNIT_ASSERT(di.Next());
            UNIT_ASSERT_EQUAL(di.Next(), nullptr);
        }
    }

    inline void TestSort() {
        TDirHier dh;
        const TString dir("tmpdir");

        //prepare fs
        {
            TMersenne<ui32> rnd;
            const TString prefixes[] = {
                "a", "b", "xxx", "111", ""};

            dh.AddDir(dir);

            for (size_t i = 0; i < 100; ++i) {
                const TString fname = dir + LOCSLASH_C + prefixes[i % Y_ARRAY_SIZE(prefixes)] + ToString(rnd.GenRand());

                dh.AddFile(fname);
            }
        }

        TVector<TString> fnames;

        {
            TDirIterator d(dir, TDirIterator::TOptions().SetSortByName());

            for (auto it = d.begin(); it != d.end(); ++it) {
                if (it->fts_info == FTS_F) {
                    fnames.push_back(it->fts_name);
                }
            }
        }

        TVector<TString> sorted(fnames);
        Sort(sorted.begin(), sorted.end());

        UNIT_ASSERT_VALUES_EQUAL(JoinWithNewline(fnames), JoinWithNewline(sorted));
    }

    inline void TestError() {
        UNIT_ASSERT_EXCEPTION(TDirIterator("./notexistingfilename"), TDirIterator::TError);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TDirIteratorTest);
