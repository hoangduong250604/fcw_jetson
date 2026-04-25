// ==============================================================================
// CUDA-MinGW Compatibility Layer
// ==============================================================================
// Provides stub implementations of MSVC-specific security functions that
// CUDA's static loader object file requires but are not available in MinGW.
// This allows CUDA to link with MinGW even if these security checks aren't used.
// ==============================================================================

// MSVC security cookie - stub for MinGW
// The security cookie is typically initialized at program startup in MSVC
// For MinGW compatibility, we provide a dummy value
extern "C" {
    // Security cookie - MSVC uses this for stack guard protection
    volatile unsigned char __security_cookie = 0xFF;
    
    // Stack protection functions - stubs for MSVC
    void __security_check_cookie(unsigned char stackcookie) {
        // Stack canary check - in MSVC, this verifies the stack guard
        // In MinGW, we just accept it without checking
        (void)stackcookie;  // Suppress unused parameter warning
    }
    
    // GSHandler (Global Static Handler) - used in exception handling
    void __GSHandlerCheck(void) {
        // Guard Stack Handler check - stub for MinGW
        // This is called during structured exception handling
    }
}

