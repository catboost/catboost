package ai.catboost.common;

import java.util.Iterator;
import java.util.function.BiPredicate;
import org.jspecify.annotations.Nullable;

public class IteratorUtils {
    public static <T extends @Nullable Object, S extends @Nullable Object> boolean elementsEqual(
        Iterator<T> lhs,
        Iterator<S> rhs,
        BiPredicate<T, S> equalFunction
    ) {
        while (lhs.hasNext() && rhs.hasNext()) {
            if (!equalFunction.test(lhs.next(), rhs.next())) {
                return false;
            }
        }
        return !lhs.hasNext() && !rhs.hasNext();
    }    
}
