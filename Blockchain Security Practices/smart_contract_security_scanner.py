def scan_contract(code):
    issues = []
    if "call.value" in code:
        issues.append("❌ Potential reentrancy attack (use Checks-Effects-Interactions pattern)")
    if "owner" not in code:
        issues.append("⚠️ Missing access control")
    if any(x in code for x in ["++", "--"]):
        issues.append("⚠️ Possible integer overflow/underflow (use SafeMath)")
    if not issues:
        issues.append("✅ No basic vulnerabilities detected")
    return issues

if __name__ == "__main__":
    sample_code = """
    function withdraw() public {
        msg.sender.call.value(balance[msg.sender])();
        balance[msg.sender] = 0;
    }
    """

    results = scan_contract(sample_code)
    print("\n📋 Security Scan Results:")
    for r in results:
        print(r)
