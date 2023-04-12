#include "shared_range.h"
#include "new.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TSharedRangeHolderPtr TSharedRangeHolder::Clone(const TSharedRangeHolderCloneOptions& /*options*/)
{
    return this;
}

std::optional<size_t> TSharedRangeHolder::GetTotalByteSize() const
{
    return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////

TSharedRangeHolderPtr MakeCompositeSharedRangeHolder(std::vector<TSharedRangeHolderPtr> holders)
{
    struct THolder
        : public TSharedRangeHolder
    {
        std::vector<TSharedRangeHolderPtr> Subholders;

        TSharedRangeHolderPtr Clone(const TSharedRangeHolderCloneOptions& options) override
        {
            auto newHolder = New<THolder>();
            newHolder->Subholders.reserve(Subholders.size());
            for (const auto& subholder : Subholders) {
                if (!subholder) {
                    continue;
                }
                if (auto clonedSubholder = subholder->Clone(options)) {
                    newHolder->Subholders.push_back(clonedSubholder);
                }
            }
            return newHolder;
        }

        std::optional<size_t> GetTotalByteSize() const override
        {
            size_t result = 0;
            for (const auto& subholder : Subholders) {
                if (!subholder) {
                    continue;
                }
                auto subsize = subholder->GetTotalByteSize();
                if (!subsize) {
                    return std::nullopt;
                }
                result += *subsize;
            }
            return result;
        }
    };

    auto holder = New<THolder>();
    holder->Subholders = std::move(holders);
    return holder;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
