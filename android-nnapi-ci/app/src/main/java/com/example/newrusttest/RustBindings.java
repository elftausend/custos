package com.example.newrusttest;

public class RustBindings {
    static {
        System.loadLibrary("nnapi-test-lib");
    }

    public static native String run(final String pattern);
}