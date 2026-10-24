#!/usr/bin/env node
/*
 * tools/ref/export_dex.js
 * Exports Gen 9 game data from a local Pokémon Showdown checkout as plain JSON for OU-JAX:
 * species, moves, items, abilities, natures and the type chart. Only entries that exist in
 * Gen 9 (isNonstandard null) are kept. For moves, items and abilities the names of their
 * event handlers (onBasePower, onModifyAtk, ...) are listed so later code can tell which
 * effects are pure data and which need hand-written logic.
 *
 * Usage: node tools/ref/export_dex.js <showdown-dir> > engine/oujax/data/dex_gen9.json
 */
'use strict';

const path = require('path');
const {execSync} = require('child_process');

const showdownDir = path.resolve(process.argv[2] || '../pokemon-showdown');
const {Dex} = require(path.join(showdownDir, 'dist', 'sim'));
const dex = Dex.forGen(9);

const available = e => e.exists && !e.isNonstandard;

function handlers(entry) {
	return Object.keys(entry)
		.filter(k => /^on[A-Z]/.test(k) && typeof entry[k] === 'function')
		.sort();
}

function plain(obj) {
	// Drops functions and undefined values, keeps nested plain data (secondary effects, flags, ...)
	return JSON.parse(JSON.stringify(obj, (k, v) => (typeof v === 'function' ? undefined : v)));
}

function conditionHandlers(entry) {
	return entry.condition ? handlers(entry.condition) : [];
}

const species = {};
for (const s of dex.species.all().filter(available)) {
	species[s.id] = {
		name: s.name,
		num: s.num,
		baseSpecies: s.baseSpecies,
		forme: s.forme,
		types: s.types,
		baseStats: s.baseStats,
		abilities: s.abilities,
		weightkg: s.weightkg,
		tier: s.tier,
		requiredItem: s.requiredItem || null,
		battleOnly: s.battleOnly || null,
		otherFormes: s.otherFormes || [],
	};
}

const MOVE_FIELDS = [
	'name', 'num', 'type', 'category', 'basePower', 'accuracy', 'pp', 'priority', 'target', 'flags',
	'critRatio', 'willCrit', 'secondary', 'secondaries', 'self', 'selfBoost', 'boosts', 'status',
	'volatileStatus', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain',
	'drain', 'recoil', 'heal', 'selfSwitch', 'forceSwitch', 'multihit', 'multiaccuracy', 'ohko',
	'damage', 'breaksProtect', 'ignoreAbility', 'ignoreDefensive', 'ignoreEvasion', 'ignoreImmunity',
	'overrideOffensiveStat', 'overrideOffensivePokemon', 'overrideDefensiveStat', 'selfdestruct',
	'mindBlownRecoil', 'struggleRecoil', 'hasCrashDamage', 'thawsTarget', 'sleepUsable', 'stallingMove',
	'hasSheerForceBoost', 'tracksTarget', 'smartTarget', 'nonGhostTarget', 'onDamagePriority',
];
const moves = {};
for (const m of dex.moves.all().filter(available)) {
	const out = {};
	for (const f of MOVE_FIELDS) if (m[f] !== undefined && m[f] !== null && m[f] !== false) out[f] = plain(m[f]);
	out.handlers = handlers(m);
	out.conditionHandlers = conditionHandlers(m);
	moves[m.id] = out;
}

const items = {};
for (const i of dex.items.all().filter(available)) {
	items[i.id] = plain({
		name: i.name,
		num: i.num,
		fling: i.fling || null,
		isBerry: !!i.isBerry,
		isChoice: !!i.isChoice,
		isGem: !!i.isGem,
		megaStone: i.megaStone || null,
		onPlate: i.onPlate || null,
		onMemory: i.onMemory || null,
		naturalGift: i.naturalGift || null,
		boosts: i.boosts || null,
		ignoreKlutz: !!i.ignoreKlutz,
		itemUser: i.itemUser || null,
		handlers: handlers(i),
		conditionHandlers: conditionHandlers(i),
	});
}

const abilities = {};
for (const a of dex.abilities.all().filter(available)) {
	abilities[a.id] = plain({
		name: a.name,
		num: a.num,
		flags: a.flags || {},
		handlers: handlers(a),
		conditionHandlers: conditionHandlers(a),
	});
}

const natures = {};
for (const n of dex.natures.all()) natures[n.id] = {name: n.name, plus: n.plus || null, minus: n.minus || null};

// damageTaken codes: 0 normal, 1 weak (x2), 2 resist (x0.5), 3 immune (x0)
const types = {};
for (const t of dex.types.all().filter(t => t.exists && !t.isNonstandard)) {
	types[t.id] = {name: t.name, damageTaken: t.damageTaken};
}

let commit = null;
try {
	commit = execSync('git rev-parse HEAD', {cwd: showdownDir}).toString().trim();
} catch (e) {}

process.stdout.write(
	JSON.stringify({showdownCommit: commit, gen: 9, species, moves, items, abilities, natures, types}) + '\n'
);
