package com.example.newrusttest;

public class RustBindings {
    static {
        System.loadLibrary("nnapitestlib");
    }

    public static native String run(final String pattern);
}