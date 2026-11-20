# Writes a fictional demo call (people, servers and words are made up) for
# screenshots and tests:  python test/make_demo.py <transcripts-dir>
import json, random, time, datetime, sys
random.seed(7)
out = sys.argv[1]
now = int(time.time() * 1000)
people = [("100000000000000001","Maya"),("100000000000000002","Jonas"),("100000000000000003","Priya"),("100000000000000004","Theo"),("100000000000000005","Ines"),("100000000000000006","Sam")]
lines = [
 "Okay, are we recording? Cool, let's start with the release plan.",
 "The build from last night passed everything except the save-file migration.",
 "I can take the migration, it's mostly the inventory schema changing.",
 "How long do you think that takes, a day?",
 "A day if nothing else breaks, maybe two with testing.",
 "Then I'd rather move the release to Thursday than rush it.",
 "Thursday works for me. Marketing only needs a week's notice.",
 "Did anyone look at the crash reports from the weekend?",
 "Yeah, most of them are the same shader cache issue on older drivers.",
 "Can we just fall back to the slower path on those cards?",
 "That's what I was thinking. It's a two line change.",
 "Let's do that and add it to the known issues list.",
 "Next thing is the audio mix, the footsteps are way too loud in the caves.",
 "I noticed that too, it gets tiring after ten minutes.",
 "I'll drop them by three decibels and send a build tonight.",
 "Also the new tutorial, people are skipping the crafting part entirely.",
 "Maybe the prompt shows up too late, after they already left the camp.",
 "We could block the gate until they craft the torch.",
 "Blocking feels heavy handed. What about a hint from the guide character?",
 "A hint is nicer. Let's try it and look at the numbers next week.",
 "I can write the dialogue for that today.",
 "Great. Anything else before we wrap up the build stuff?",
 "Just that the Linux build is still failing on the CI runner.",
 "It's the new compiler version, I pinned it on my branch.",
 "Merge it after the call and I'll rerun the pipeline.",
 "Okay, switching topics, the community stream on Saturday.",
 "Who's hosting? I can do the first hour.",
 "I'll do the second hour and the giveaway.",
 "We need the patch notes finished by Friday for that.",
 "I'll draft them tomorrow morning and share the doc.",

 "One more thing on the build, the installer size jumped by forty megabytes.",
 "That's the uncompressed voice lines, I forgot to run the packer.",
 "Can you add the packer step to the pipeline so it can't happen again?",
 "Sure, I'll make the build fail if the assets aren't packed.",
 "Nice. Priya, how's the localization going?",
 "German and French are done, Japanese comes back from the agency Monday.",
 "Do we ship without Japanese or wait?",
 "Ship without it and patch it in, the store page doesn't promise it yet.",
 "Agreed. I'll hide the language option until then.",
 "On the stream, should we show the new boss fight or keep it secret?",
 "Show the first phase only, keep the second phase as a surprise.",
 "I like that. I'll set up a save right before the arena.",
 "Make sure the debug overlay is off this time.",
 "Ha, yes, last stream half the chat was asking about the frame counter.",
 "Theo, can you check the controller remapping bug before Thursday?",
 "Already on it, it only happens when two controllers are plugged in.",
 "Rare enough to ship with, but let's mention it in the notes.",
 "Last thing from me, the trailer music license is signed.",
 "Great news. Send me the final file and I'll cut the thirty second version.",
 "Will do, end of day.",
 "Alright, same time next week. Thanks all.",
]
t = now - 13 * 60000
recs = []
w = [5, 4, 3, 3, 2, 1]
for i in range(150):
    uid, name = random.choices(people, weights=w)[0]
    text = lines[i % len(lines)]
    d = int(len(text.split()) * 330 + random.randint(200, 900))
    if random.random() < 0.25: t -= random.randint(300, 1200)
    recs.append({"t": t, "end": t + d, "user": uid, "speaker": name, "text": text,
                 "ch": "200000000000000001", "chName": "Weekly sync", "guild": "300000000000000001", "guildName": "Lantern Games"})
    t += d + random.randint(1500, 7000)
    if t > now - 5000: break
earlier = []
t2 = now - 5 * 3600000
for i in range(40):
    uid, name = random.choice(people[:3])
    text = random.choice(lines)
    earlier.append({"t": t2, "end": t2 + 3000, "user": uid, "speaker": name, "text": text, "ch": "200000000000000002", "chName": "Design review", "guild": "300000000000000001", "guildName": "Lantern Games"})
    t2 += 6000
t3 = now - 3 * 3600000
for i in range(30):
    uid, name = random.choice(people[3:])
    earlier.append({"t": t3, "end": t3 + 3000, "user": uid, "speaker": name, "text": random.choice(lines), "ch": "200000000000000003", "chName": "General", "guild": "300000000000000002", "guildName": "Book Club"})
    t3 += 7000
day = datetime.date.today().isoformat()
with open(f"{out}/{day}.jsonl", "w") as f:
    for r in sorted(earlier + recs, key=lambda r: r["t"]):
        f.write(json.dumps(r) + "\n")
print(len(recs), "lines")
