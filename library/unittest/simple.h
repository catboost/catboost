#pragma once

#include "registar.h"

namespace NUnitTest {
    struct TSimpleTestExecutor: public TTestBase {
        typedef yvector<TTest> TTests;

        TTests Tests;

        virtual void Execute() override final {
            AtStart();

            for (typename TTests::iterator i = Tests.begin(), ie = Tests.end(); i != ie; ++i) {
                if (!CheckAccessTest(i->Name)) {
                    continue;
                }
                TTestContext context;
                try {
                    BeforeTest(i->Name);
                    {
                        TCleanUp cleaner(this);
                        TTestBase::Run([i, &context] { i->Body(context); }, Name(), i->Name, i->ForceFork);
                    }
                } catch (const ::NUnitTest::TAssertException&) {
                } catch (const yexception& e) {
                    CATCH_REACTION_BT(i->Name, e, &context);
                } catch (const std::exception& e) {
                    CATCH_REACTION(i->Name, e, &context);
                } catch (...) {
                    AddError("non-std exception!", &context);
                }
                Finish(i->Name, &context);
            }

            AtEnd();
        }
    };
}
