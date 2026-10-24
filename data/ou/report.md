# Gen 9 OU data build report

Built 2026-10-24T21:38:32Z against Showdown `c0845f33d381402dfcb18ce324c20a94f18af075` (format `gen9ou`). Stats months: 2026-09, 2026-08, 2026-07; recency reference month 2026-09.

## Sources

| Source | Candidates | Valid | Dropped | In pool after dedup |
|---|---|---|---|---|
| metamon/hl_05_26 | 43764 | 43764 | 0 | 43696 |
| metamon/competitive | 16 | 11 | 5 | 11 |

Metamon team files are plain Showdown exports: they do not mark which set fields were revealed in the replay and which were predicted from usage stats, and carry no per-team rating or date. `revealed_fraction` and `rating` are therefore null; `hl_05_26` is dated 2026-05 from its name, the expert `competitive` set is undated.

Other sources (no teams, used for set archetypes and marginals): `chaos/2026-09/1695`, `chaos/2026-09/1825`, `chaos/2026-08/1695`, `chaos/2026-08/1825`, `chaos/2026-07/1695`, `chaos/2026-07/1825`, `smogon/analysis_sets`, `showdown/factory_sets_ou`

## Validation

Every candidate team was validated with Showdown's `TeamValidator` for `gen9ou` (in-process, batched). Invalid teams are dropped:

| Reason | Teams | Example |
|---|---|---|
| banned | 5 | metamon/competitive/team_0.gen9ou_team: Kyurem's move Tera Blast is banned. |

Deduplication removed 0 exact and 68 near duplicates (same six species, >= 90% identical set fields). **Pool size: 43707 teams**, total weight 17375.4 (weight = recency weight x copies).

## Archetypes

| Tag | Teams | Share of pool weight |
|---|---|---|
| hazard_removal | 29166 | 66.7% |
| balance | 20330 | 46.5% |
| hazard_stack | 12052 | 27.6% |
| hyper_offence | 11094 | 25.4% |
| offence | 11027 | 25.2% |
| volt_turn | 9821 | 22.5% |
| snow | 5887 | 13.5% |
| sand | 3243 | 7.4% |
| grassy_terrain | 2829 | 6.5% |
| sun | 2692 | 6.2% |
| webs | 2207 | 5.1% |
| rain | 2185 | 5.0% |
| stall | 1256 | 2.9% |
| trick_room | 594 | 1.4% |
| electric_terrain | 333 | 0.8% |
| psychic_terrain | 311 | 0.7% |
| misty_terrain | 48 | 0.1% |

## Top 30 species by pool weight

| # | Species | Share of teams |
|---|---|---|
| 1 | greattusk | 33.7% |
| 2 | kingambit | 21.4% |
| 3 | gholdengo | 21.1% |
| 4 | ragingbolt | 19.8% |
| 5 | zamazenta | 17.0% |
| 6 | ogerponwellspring | 15.8% |
| 7 | ironvaliant | 15.1% |
| 8 | dragonite | 15.0% |
| 9 | irontreads | 14.8% |
| 10 | slowkinggalar | 14.1% |
| 11 | corviknight | 13.6% |
| 12 | tinglu | 13.3% |
| 13 | hatterene | 13.2% |
| 14 | cinderace | 12.5% |
| 15 | pecharunt | 12.2% |
| 16 | dragapult | 11.3% |
| 17 | alomomola | 11.0% |
| 18 | kyurem | 10.6% |
| 19 | samurotthisui | 10.4% |
| 20 | gliscor | 9.4% |
| 21 | darkrai | 8.9% |
| 22 | zapdos | 8.7% |
| 23 | landorustherian | 8.5% |
| 24 | ceruledge | 8.1% |
| 25 | ironmoth | 7.7% |
| 26 | tyranitar | 7.2% |
| 27 | rillaboom | 6.5% |
| 28 | glimmora | 6.4% |
| 29 | tornadustherian | 6.4% |
| 30 | walkingwake | 6.4% |

## Meta coverage

520 moves, 136 items and 223 abilities appear in the pool or chaos stats. Smallest greedy sets (elements ranked by usage-weighted pool share) that fully cover a share of usage-weighted pool teams:

| Target | Covered | Moves | Items | Abilities |
|---|---|---|---|---|
| 95% | 95.02% | 336 | 64 | 138 |
| 99% | 99.01% | 395 | 81 | 181 |

Per Pokémon set instead (a set is covered when its moves, item and ability are all in), taking prefixes of the same ranking:

| Target | Covered | Moves | Items | Abilities |
|---|---|---|---|---|
| 95% of sets | 95.05% | 247 | 49 | 88 |
| 99% of sets | 99.00% | 332 | 64 | 132 |

### 95% coverage set

**Moves (336):** knockoff, stealthrock, uturn, earthquake, swordsdance, rapidspin, shadowball, calmmind, closecombat, roost, recover, icespinner, suckerpunch, headlongrush, moonblast, bodypress, dracometeor, ironhead, flipturn, earthpower, protect, encore, thunderwave, willowisp, thunderbolt, icebeam, flamethrower, voltswitch, psychicnoise, spikes, sludgebomb, nastyplot, ivycudgel, lowkick, dragondance, kowtowcleave, substitute, thunderclap, trick, makeitrain, dragonpulse, toxic, irondefense, taunt, hurricane, crunch, defog, darkpulse, drainingkiss, focusblast, dazzlinggleam, roar, hex, stoneedge, psyshock, weatherball, healingwish, bulkup, pyroball, extremespeed, futuresight, chillyreception, bravebird, surf, ruination, wish, ceaselessedge, partingshot, scald, malignantchain, aquajet, courtchange, heavyslam, nuzzle, freezedry, heatwave, playrough, powerwhip, drainpunch, whirlwind, shadowsneak, sludgewave, liquidation, rest, dragondarts, icepunch, foulplay, scaleshot, hornleech, fierydance, painsplit, mysticalfire, razorshell, firepunch, bitterblade, sleeptalk, mirrorcoat, curse, energyball, poltergeist, grassyglide, gunkshot, gigadrain, lightscreen, tripleaxel, discharge, bleakwindstorm, toxicspikes, saltcure, overheat, flashcannon, vacuumwave, reflect, iciclespear, stickyweb, poisonjab, moonlight, headsmash, trickroom, hydrosteam, psychic, trailblaze, firstimpression, mortalspin, highjumpkick, bulletpunch, storedpower, hydropump, dragontail, woodhammer, facade, agility, superpower, destinybond, icefang, powergem, magmastorm, strengthsap, aquacutter, seismictoss, psychoboost, tachyoncutter, meteorbeam, leechlife, softboiled, rocktomb, matchagotcha, outrage, strangesteam, flareblitz, aurasphere, thunder, iceshard, waterfall, lavaplume, ficklebeam, blizzard, grassknot, steelbeam, sacredsword, spiritbreak, steameruption, haze, rockslide, auroraveil, endeavor, stunspore, tickle, flowertrick, solarbeam, supercellslam, growth, acrobatics, leafstorm, scorchingsands, payback, gigatonhammer, fireblast, phantomforce, skillswap, hyperspacefury, bodyslam, synthesis, psychicfangs, airslash, wavecrash, avalanche, iciclecrash, dualwingbeat, machpunch, highhorsepower, wildcharge, slackoff, banefulbunker, morningsun, leechseed, whirlpool, lunge, endure, revivalblessing, icehammer, clearsmog, lunardance, expandingforce, lusterpurge, alluringvoice, bellydrum, populationbomb, tidyup, tailglow, dragonclaw, hypervoice, watershuriken, solarblade, acidarmor, aquastep, secretsword, infestation, glare, thunderpunch, beatup, torchsong, zenheadbutt, takeheart, firefang, eruption, bite, doubleshock, bugbuzz, shadowclaw, memento, rockblast, shellsmash, snowscape, throatchop, spikyshield, bulletseed, clangoroussoul, smackdown, brickbreak, temperflare, explosion, mightycleave, cosmicpower, quiverdance, fakeout, amnesia, chillingwater, tailwind, crabhammer, sparklingaria, doubleedge, psychocut, fierywrath, healbell, stompingtantrum, icywind, sacredfire, quickattack, nightslash, armorcannon, psyblade, boomburst, stoneaxe, firespin, leafblade, clangingscales, transform, hyperbeam, switcheroo, bulldoze, teleport, photongeyser, thunderouskick, focusenergy, imprison, seedbomb, psychicterrain, flamecharge, junglehealing, perishsong, blazekick, mistball, metronome, counter, beakblast, nightshade, electroball, axekick, upperhand, mistyterrain, ancientpower, risingvoltage, swagger, magnetrise, powertrip, extrasensory, xscissor, triattack, mistyexplosion, doomdesire, poisonfang, accelerock, fling, meteormash, howl, snarl, acidspray, finalgambit, diamondstorm, detect, swift, victorydance, gyroball, meanlook, megahorn, barbbarrage, breakingswipe, gravity, waterpulse, chargebeam, drillrun, raindance, thief, bugbite, sandsearstorm, reversal, burningjealousy, shiftgear, revelationdance, sunnyday, shellsidearm, hammerarm, ragingbull, dragonenergy, block

**Items (64):** leftovers, heavydutyboots, rockyhelmet, boosterenergy, lifeorb, assaultvest, airballoon, choiceband, choicespecs, choicescarf, wellspringmask, focussash, toxicorb, heatrock, lightclay, ejectbutton, loadeddice, damprock, blackglasses, redcard, cornerstonemask, weaknesspolicy, blacksludge, flameorb, grassyseed, powerherb, custapberry, sitrusberry, terrainextender, smoothrock, lumberry, covertcloak, ejectpack, shucaberry, leppaberry, colburberry, widelens, souldew, expertbelt, chestoberry, eviolite, wiseglasses, metalcoat, throatspray, mentalherb, mirrorherb, silkscarf, stickybarb, psychicseed, electricseed, whiteherb, mysticwater, nevermeltice, scopelens, silverpowder, quickclaw, icyrock, clearamulet, salacberry, keeberry, magnet, blunderpolicy, mistyseed, punchingglove

**Abilities (138):** protosynthesis, quarkdrive, regenerator, pressure, goodasgold, supremeoverlord, waterabsorb, dauntlessshield, multiscale, magicbounce, vesselofruin, poisonpuppeteer, levitate, sharpness, libero, intimidate, poisonheal, baddreams, static, unaware, weakarmor, flamebody, sandstream, infiltrator, grassysurge, toxicdebris, sturdy, technician, drought, defiant, purifyingsalt, clearbody, flashfire, drizzle, neutralizinggas, contrary, naturalcure, swiftswim, roughskin, magicguard, sandrush, magician, tintedlens, heatproof, protean, moldbreaker, torrent, guts, liquidvoice, waterbubble, prankster, snowwarning, chlorophyll, blaze, shielddust, overcoat, hydration, toxicchain, voltabsorb, synchronize, unburden, moxie, pickpocket, speedboost, justified, bulletproof, cursedbody, battlebond, rockhead, innerfocus, hugepower, dryskin, ironfist, trace, sapsipper, leafguard, electricsurge, gooey, psychicsurge, disguise, serenegrace, adaptability, tabletsofruin, poisontouch, competitive, transistor, triage, guarddog, mirrorarmor, cutecharm, thickfat, berserk, bigpecks, skilllink, magnetpull, soundproof, prismarmor, galewings, slushrush, raindish, reckless, download, solarpower, oblivious, liquidooze, corrosion, quickdraw, illusion, gulpmissile, overgrow, solidrock, shellarmor, sheerforce, windrider, effectspore, toughclaws, myceliummight, megalauncher, sniper, owntempo, healer, mistysurge, swarm, dancer, pixilate, hospitality, icescales, scrappy, cloudnine, dragonsmaw, stakeout, icebody, dazzling, damp, harvest, frisk, sweetveil, stormdrain

### 99% coverage set

**Moves (395):** knockoff, stealthrock, uturn, earthquake, swordsdance, rapidspin, shadowball, calmmind, closecombat, roost, recover, icespinner, suckerpunch, headlongrush, moonblast, bodypress, dracometeor, ironhead, flipturn, earthpower, protect, encore, thunderwave, willowisp, thunderbolt, icebeam, flamethrower, voltswitch, psychicnoise, spikes, sludgebomb, nastyplot, ivycudgel, lowkick, dragondance, kowtowcleave, substitute, thunderclap, trick, makeitrain, dragonpulse, toxic, irondefense, taunt, hurricane, crunch, defog, darkpulse, drainingkiss, focusblast, dazzlinggleam, roar, hex, stoneedge, psyshock, weatherball, healingwish, bulkup, pyroball, extremespeed, futuresight, chillyreception, bravebird, surf, ruination, wish, ceaselessedge, partingshot, scald, malignantchain, aquajet, courtchange, heavyslam, nuzzle, freezedry, heatwave, playrough, powerwhip, drainpunch, whirlwind, shadowsneak, sludgewave, liquidation, rest, dragondarts, icepunch, foulplay, scaleshot, hornleech, fierydance, painsplit, mysticalfire, razorshell, firepunch, bitterblade, sleeptalk, mirrorcoat, curse, energyball, poltergeist, grassyglide, gunkshot, gigadrain, lightscreen, tripleaxel, discharge, bleakwindstorm, toxicspikes, saltcure, overheat, flashcannon, vacuumwave, reflect, iciclespear, stickyweb, poisonjab, moonlight, headsmash, trickroom, hydrosteam, psychic, trailblaze, firstimpression, mortalspin, highjumpkick, bulletpunch, storedpower, hydropump, dragontail, woodhammer, facade, agility, superpower, destinybond, icefang, powergem, magmastorm, strengthsap, aquacutter, seismictoss, psychoboost, tachyoncutter, meteorbeam, leechlife, softboiled, rocktomb, matchagotcha, outrage, strangesteam, flareblitz, aurasphere, thunder, iceshard, waterfall, lavaplume, ficklebeam, blizzard, grassknot, steelbeam, sacredsword, spiritbreak, steameruption, haze, rockslide, auroraveil, endeavor, stunspore, tickle, flowertrick, solarbeam, supercellslam, growth, acrobatics, leafstorm, scorchingsands, payback, gigatonhammer, fireblast, phantomforce, skillswap, hyperspacefury, bodyslam, synthesis, psychicfangs, airslash, wavecrash, avalanche, iciclecrash, dualwingbeat, machpunch, highhorsepower, wildcharge, slackoff, banefulbunker, morningsun, leechseed, whirlpool, lunge, endure, revivalblessing, icehammer, clearsmog, lunardance, expandingforce, lusterpurge, alluringvoice, bellydrum, populationbomb, tidyup, tailglow, dragonclaw, hypervoice, watershuriken, solarblade, acidarmor, aquastep, secretsword, infestation, glare, thunderpunch, beatup, torchsong, zenheadbutt, takeheart, firefang, eruption, bite, doubleshock, bugbuzz, shadowclaw, memento, rockblast, shellsmash, snowscape, throatchop, spikyshield, bulletseed, clangoroussoul, smackdown, brickbreak, temperflare, explosion, mightycleave, cosmicpower, quiverdance, fakeout, amnesia, chillingwater, tailwind, crabhammer, sparklingaria, doubleedge, psychocut, fierywrath, healbell, stompingtantrum, icywind, sacredfire, quickattack, nightslash, armorcannon, psyblade, boomburst, stoneaxe, firespin, leafblade, clangingscales, transform, hyperbeam, switcheroo, bulldoze, teleport, photongeyser, thunderouskick, focusenergy, imprison, seedbomb, psychicterrain, flamecharge, junglehealing, perishsong, blazekick, mistball, metronome, counter, beakblast, nightshade, electroball, axekick, upperhand, mistyterrain, ancientpower, risingvoltage, swagger, magnetrise, powertrip, extrasensory, xscissor, triattack, mistyexplosion, doomdesire, poisonfang, accelerock, fling, meteormash, howl, snarl, acidspray, finalgambit, diamondstorm, detect, swift, victorydance, gyroball, meanlook, megahorn, barbbarrage, breakingswipe, gravity, waterpulse, chargebeam, drillrun, raindance, thief, bugbite, sandsearstorm, reversal, burningjealousy, shiftgear, revelationdance, sunnyday, shellsidearm, hammerarm, ragingbull, dragonenergy, block, jawlock, rockpolish, esperwing, stockpile, infernalparade, electricterrain, speedswap, aquatail, bittermalice, magicroom, eeriespell, pounce, drillpeck, focuspunch, honeclaws, tailslap, lifedew, skyattack, rockwrecker, burningbulwark, thunderfang, zingzap, charge, spite, mysticalpower, circlethrow, fellstinger, cottonguard, flyingpress, lunarblessing, disable, thundercage, zapcannon, charm, noretreat, lowsweep, overdrive, worryseed, snipeshot, triplearrows, muddywater, mountaingale, petalblizzard, darkestlariat, poisonsting, copycat, doublekick, gigaimpact, fairylock, relicsong, heatcrash, sandtomb, aerialace, flail, ingrain, freezingglare, shoreup, lashout, sandstorm

**Items (81):** leftovers, heavydutyboots, rockyhelmet, boosterenergy, lifeorb, assaultvest, airballoon, choiceband, choicespecs, choicescarf, wellspringmask, focussash, toxicorb, heatrock, lightclay, ejectbutton, loadeddice, damprock, blackglasses, redcard, cornerstonemask, weaknesspolicy, blacksludge, flameorb, grassyseed, powerherb, custapberry, sitrusberry, terrainextender, smoothrock, lumberry, covertcloak, ejectpack, shucaberry, leppaberry, colburberry, widelens, souldew, expertbelt, chestoberry, eviolite, wiseglasses, metalcoat, throatspray, mentalherb, mirrorherb, silkscarf, stickybarb, psychicseed, electricseed, whiteherb, mysticwater, nevermeltice, scopelens, silverpowder, quickclaw, icyrock, clearamulet, salacberry, keeberry, magnet, blunderpolicy, mistyseed, punchingglove, miracleseed, bigroot, protectivepads, metronome, softsand, normalgem, ironball, muscleband, tangaberry, dragonfang, liechiberry, roseliberry, occaberry, razorclaw, marangaberry, charcoal, laggingtail

**Abilities (181):** protosynthesis, quarkdrive, regenerator, pressure, goodasgold, supremeoverlord, waterabsorb, dauntlessshield, multiscale, magicbounce, vesselofruin, poisonpuppeteer, levitate, sharpness, libero, intimidate, poisonheal, baddreams, static, unaware, weakarmor, flamebody, sandstream, infiltrator, grassysurge, toxicdebris, sturdy, technician, drought, defiant, purifyingsalt, clearbody, flashfire, drizzle, neutralizinggas, contrary, naturalcure, swiftswim, roughskin, magicguard, sandrush, magician, tintedlens, heatproof, protean, moldbreaker, torrent, guts, liquidvoice, waterbubble, prankster, snowwarning, chlorophyll, blaze, shielddust, overcoat, hydration, toxicchain, voltabsorb, synchronize, unburden, moxie, pickpocket, speedboost, justified, bulletproof, cursedbody, battlebond, rockhead, innerfocus, hugepower, dryskin, ironfist, trace, sapsipper, leafguard, electricsurge, gooey, psychicsurge, disguise, serenegrace, adaptability, tabletsofruin, poisontouch, competitive, transistor, triage, guarddog, mirrorarmor, cutecharm, thickfat, berserk, bigpecks, skilllink, magnetpull, soundproof, prismarmor, galewings, slushrush, raindish, reckless, download, solarpower, oblivious, liquidooze, corrosion, quickdraw, illusion, gulpmissile, overgrow, solidrock, shellarmor, sheerforce, windrider, effectspore, toughclaws, myceliummight, megalauncher, sniper, owntempo, healer, mistysurge, swarm, dancer, pixilate, hospitality, icescales, scrappy, cloudnine, dragonsmaw, stakeout, icebody, dazzling, damp, harvest, frisk, sweetveil, stormdrain, armortail, merciless, seedsower, curiousmedicine, filter, analytic, punkrock, flowerveil, keeneye, fluffy, strongjaw, aftermath, honeygather, vitalspirit, poisonpoint, chillingneigh, comatose, eartheater, queenlymajesty, truant, marvelscale, stickyhold, slowstart, aromaveil, compoundeyes, lightningrod, surgesurfer, waterveil, electromorphosis, shedskin, sandforce, galvanize, watercompaction, hustle, insomnia, rattled, hypercutter, unnerve, shieldsdown, steadfast, quickfeet, limber, lightmetal

## Evaluation teams

| File | Source | Tags |
|---|---|---|
| 01_balance_gliscor-alomomola.txt | metamon/competitive | balance, hazard_removal, hazard_stack, volt_turn |
| 02_stall_weavile-dragonite.txt | metamon/competitive | hazard_stack, stall |
| 03_balance_tinkaton-greattusk.txt | metamon/competitive | balance, hazard_removal |
| 04_balance_gholdengo-darkrai.txt | metamon/competitive | balance, hazard_stack |
| 05_balance_ogerponwellspring-garganacl.txt | metamon/competitive | balance, hazard_removal, hazard_stack |
| 06_balance_lokix-pecharunt.txt | metamon/competitive | balance, hazard_stack, volt_turn |
| 07_balance_tinglu-corviknight.txt | metamon/competitive | balance, hazard_removal, snow, volt_turn |
| 08_balance_dondozo-weezinggalar.txt | metamon/competitive | balance, hazard_removal |
| 09_balance_tinglu-weezinggalar.txt | metamon/competitive | balance, hazard_removal |
| 10_balance_zapdos-hydrapple.txt | metamon/competitive | balance, snow |
| 11_hyper_offence_pelipper-manaphy.txt | metamon/hl_05_26 | hyper_offence, rain |
| 12_hyper_offence_charizard-walkingwake.txt | metamon/hl_05_26 | hazard_removal, hyper_offence, sun |
| 13_balance_tinkaton-tyranitar.txt | metamon/hl_05_26 | balance, hazard_removal, hazard_stack, sand |
| 14_offence_gholdengo-ragingbolt.txt | metamon/hl_05_26 | offence, webs |
| 15_balance_hatterene-ursaluna.txt | metamon/hl_05_26 | balance, trick_room |
| 16_balance_tinkaton-ogerpon.txt | metamon/hl_05_26 | balance, hazard_removal |
| 17_balance_ironmoth-umbreon.txt | metamon/hl_05_26 | balance, hazard_removal, hazard_stack |
| 18_balance_lokix-ironvaliant.txt | metamon/hl_05_26 | balance, hazard_removal, volt_turn |
| 19_offence_flamigo-landorustherian.txt | metamon/hl_05_26 | hazard_stack, offence, volt_turn |
| 20_balance_pelipper-hatterene.txt | metamon/hl_05_26 | balance, hazard_removal, rain, volt_turn |
