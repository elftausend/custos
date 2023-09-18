package com.example.newrusttest

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.newrusttest", appContext.packageName)
    }

    @Test
    fun testAdd() {
        assertEquals(4, 2+2)
    }

    @Test
    fun testRust() {
        assertEquals("[2, 8, 18, 32, 50, 72, 98, 128, 162, 210]", RustBindings.run("out"));
    }
}