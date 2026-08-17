/**
 * @file test_suite.cpp
 * @brief Suite de Pruebas Unitarias Numéricas para verificar la matemática de NeuralSuite en C++.
 */

#include "tensor.h"
#include "tokenizer.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_matmul() {
    std::cout << "🧪 [Test 1] Multiplicación de Matrices (GEMM)... ";
    ns::Tensor A({2, 3});
    ns::Tensor B({3, 2});
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A.data[0] = 1; A.data[1] = 2; A.data[2] = 3;
    A.data[3] = 4; A.data[4] = 5; A.data[5] = 6;

    // B = [[7, 8], [9, 1], [2, 3]]
    B.data[0] = 7; B.data[1] = 8;
    B.data[2] = 9; B.data[3] = 1;
    B.data[4] = 2; B.data[5] = 3;

    ns::Tensor C;
    ns::matmul(A, B, C);

    // C = [[31, 19], [85, 55]]
    assert(std::abs(C.data[0] - 31.0f) < 1e-4f);
    assert(std::abs(C.data[1] - 19.0f) < 1e-4f);
    assert(std::abs(C.data[2] - 85.0f) < 1e-4f);
    assert(std::abs(C.data[3] - 55.0f) < 1e-4f);

    std::cout << "PASADO ✅\n";
}

void test_layernorm() {
    std::cout << "🧪 [Test 2] Normalización de Capa (LayerNorm)... ";
    ns::Tensor x({1, 4});
    x.data[0] = 2.0f; x.data[1] = 4.0f; x.data[2] = 4.0f; x.data[3] = 6.0f;

    ns::Tensor gamma({4}); gamma.ones();
    ns::Tensor beta({4}); beta.zeros();

    ns::Tensor out, mean, rstd;
    ns::layernorm_forward(x, gamma, beta, out, mean, rstd);

    // Media = (2+4+4+6)/4 = 4.0
    assert(std::abs(mean.data[0] - 4.0f) < 1e-4f);

    std::cout << "PASADO ✅\n";
}

void test_tokenizer() {
    std::cout << "🧪 [Test 3] Tokenizador de Caracteres C++... ";
    std::string sample = "Hello C++!";
    ns::CharTokenizer tok(sample);
    
    std::vector<int> encoded = tok.encode(sample);
    std::string decoded = tok.decode(encoded);

    assert(sample == decoded);
    std::cout << "PASADO ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "🚀 Ejecutando Suite de Pruebas Unitarias de NeuralSuite (C++)\n";
    std::cout << "============================================================\n";

    test_matmul();
    test_layernorm();
    test_tokenizer();

    std::cout << "============================================================\n";
    std::cout << "✅ ¡Todas las pruebas unitarias numéricas pasaron con éxito!\n";
    std::cout << "============================================================\n";
    return 0;
}
