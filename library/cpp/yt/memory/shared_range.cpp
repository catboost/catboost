#include "shared_range.h"
#include "new.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TSharedRangeHolderPtr TSharedRangeHolder::Clone(const TSharedRangeHolderCloneOptions& /*options*/)
{
    return this;
}

////////////////////////////////////////////////////////////////////////////////

TSharedRangeHolderPtr MakeCompositeSharedRangeHolder(std::vector<TSharedRangeHolderPtr> holders)
{
    struct THolder
        : public TSharedRangeHolder
    {
        std::vector<TSharedRangeHolderPtr> Holders;

        TSharedRangeHolderPtr Clone(const TSharedRangeHolderCloneOptions& options) override
        {
            auto newHolder = New<THolder>();
            newHolder->Holders.reserve(Holders.size());
            for (const auto& holder : Holders) {
                if (!holder) {
                    continue;
                }
                if (auto cloned = holder->Clone(options)) {
                    newHolder->Holders.push_back(cloned);
                }
            }

            return newHolder;
        }
    };

    auto holder = New<THolder>();
    holder->Holders = std::move(holders);
    return holder;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
