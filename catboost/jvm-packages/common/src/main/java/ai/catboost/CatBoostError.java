package ai.catboost;

public class CatBoostError extends Exception {
    private static final long serialVersionUID = 1L;

    public CatBoostError(String message) {
        super(message);
    }
}
