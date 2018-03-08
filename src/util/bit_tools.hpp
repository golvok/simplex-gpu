#ifndef UTIL__BIT_TOOLS_H
#define UTIL__BIT_TOOLS_H

#include <type_traits>

namespace util {

	template<typename DEST, typename SRC>
	DEST no_sign_ext_cast(const SRC src) {
		return static_cast<DEST>(static_cast<std::make_unsigned_t<SRC>>(src));
	}

}

#endif // UTIL__BIT_TOOLS_H
