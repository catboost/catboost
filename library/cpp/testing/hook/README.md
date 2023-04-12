# Hook for google benchmark and gtest

Y_TEST_HOOK_BEFORE_INIT - вызывается перед инициализацией соответствующего фреймворка
Y_TEST_HOOK_BEFORE_RUN - вызывается перед запуском тестов
Y_TEST_HOOK_AFTER_RUN - вызывается всегда после завершения выполнения тестов, 
                        если этап инициализации был успешным
                        
## Примеры:

```
Y_TEST_HOOK_BEFORE_INIT(SetupMyApp) {
  // ваш код для выполнения перед инициализацией фреймворка
}

Y_TEST_HOOK_BEFORE_RUN(InitMyApp) {
  // ваш код для выполнения перед запуском тестов
}                        

Y_TEST_HOOK_AFTER_RUN(CleanMyApp) {
  // ваш код для выполнения после завершения тестов
}
```

## Тесты:
тесты лаунчерах соотвествующих фреймворков (gtest, gbenchmark и unittest)
