### Anything that gets preprocessed can hide instructions!!
Users think they're uploading one image, but the model sees something completely different after scaling. The mismatch creates opportunities for hidden instructions, malicious text, or deceptive payloads. Image scaling is just one representative example of this class and can be extended to audio!

#### 1. Agentic systems amplify the damage
Agentic systems wired into calendars, email, search, or file systems, can accidentally act on hidden instructions:

- Exfiltrating data
- Sending Unauthorized Emails
- Scheduling or Modifying Calendar Events
- Performing unintended tool calls like Unauthorized File Operations
- Triggering Costly API Calls or Actions
- Inserting Malicious Code Snippets
- Spread False Identities

The model's "innocent" image input becomes a control channel.

#### 2. Downscaling will continue to be common
Even as models evolve, UIs, mobile apps, and backend services often resize images for compatibility or latency reasons. That makes the attack surface persistent.

The dangerous part isn't that the model is "weak to injections", it's just doing it's job.

This reframes the risk: secure AI != secure model; secure AI = secure orchestration.

### Root Cause
#### Lack of user visibility
- If users could see the downscaled version, 90% of these attacks lose their stealth.
- Mismatched user-model view = blind spot = vulnerability.