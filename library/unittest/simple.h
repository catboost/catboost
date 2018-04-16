#pragma once

#include "registar.h"

namespace NUnitTest {
    struct TSimpleTestExecutor: public TTestBase {
        typedef TVector<TBaseTestCase> TTests;

        TTests Tests;

        virtual void Execute() override final {
            AtStart();

            for (typename TTests::iterator i = Tests.begin(), ie = Tests.end(); i != ie; ++i) {
                if (!CheckAccessTest(i->Name_)) {
                    continue;
                }
                TTestContext context(this->Processor());
                try {
                    BeforeTest(i->Name_);
                    {
                        TCleanUp cleaner(this);
                        TTestBase::Run([i, &context] { i->Body_(context); }, Name(), i->Name_, i->ForceFork_);
                    }
                } catch (const ::NUnitTest::TAssertException&) {
                } catch (const yexception& e) {
                    CATCH_REACTION_BT(i->Name_, e, &context);
                } catch (const std::exception& e) {
                    CATCH_REACTION(i->Name_, e, &context);
                } catch (...) {
                    AddError("non-std exception!", &context);
                }
                Finish(i->Name_, &context);
            }

            AtEnd();
        }
    };
}
