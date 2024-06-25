#include "fsm.h"
#include "defs.h"

namespace Pire {
	class HalfFinalFsm {
	public:
		HalfFinalFsm(const Fsm& sourceFsm) : fsm(sourceFsm) {}

		void MakeScanner();

		/// Non greedy counter without allowed intersects works correctly on all regexps
		/// Non simplified non greedy counter with allowed intersects counts number of positions in string,
		/// on which ends at least one substring that matches regexp
		/// Simplified non greedy counter with allowed intersects does not always work correctly,
		/// but has fewer number of states and more regexps can be glued into single scanner
		void MakeNonGreedyCounter(bool allowIntersects = true, bool simplify = true);

		// Simplified counter does not work correctly on all regexps, but has less number of states
		// and allows to glue larger number of scanners into one within the same size limit
		void MakeGreedyCounter(bool simplify = true);

		const Fsm& GetFsm() const { return fsm; }

		template<class Scanner>
		Scanner Compile() const;

		size_t GetCount(size_t state) const;

		size_t GetTotalCount() const;

		static size_t MaxCountDepth;
	private:
		Fsm fsm;

		bool AllowHalfFinals();

		void MakeHalfFinal();

		void DisconnectFinals(bool allowIntersects);

		void Determine(size_t depth = MaxCountDepth);
	};

	template<class Scanner>
	Scanner HalfFinalFsm::Compile() const {
		auto scanner = Scanner(*this);
	}
}
