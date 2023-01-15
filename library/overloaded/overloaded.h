#pragma once

template <class... Fs>
struct TOverloaded : Fs... {
    using Fs::operator()...;
};

template <class... Fs>
TOverloaded(Fs...) -> TOverloaded<Fs...>;
