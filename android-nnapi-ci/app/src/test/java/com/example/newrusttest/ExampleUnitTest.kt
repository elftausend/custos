package com.example.newrusttest

import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        RustBindings.greeting("hi");
        assertEquals(4, 2 + 2)
    }

    @Test
    fun addition_isWrong() {
        assertEquals(5, 3 + 2)
    }

    @Test
    fun addition_isWrong2() {
        assertEquals(5, 3 + 2)
    }
}