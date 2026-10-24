#!/usr/bin/env node
/*
 * tools/ref/ref.js
 * Reference harness for OU-JAX: Pokémon Showdown's own simulator and @smogon/calc behind a
 * JSON-lines protocol (one request per stdin line, one response per stdout line).
 *
 * Usage: node tools/ref/ref.js <showdown-dir>
 *
 * Requests ({"id": any, "cmd": ..., ...}); every response echoes "id" and has "ok":
 *   {"cmd": "pack", "teams": [export or packed team, ...]} -> {"packed": [packed team, ...]}
 *   {"cmd": "stats", "teams": [packed team, ...]}
 *       -> {"stats": [[{species, maxhp, stats: {atk, def, spa, spd, spe}}, ...], ...]}
 *   {"cmd": "calc", "attacker": P, "defender": P, "move": M, "field": F}
 *       P: {species, level?, item?, ability?, nature?, evs?, ivs?, boosts?, status?, teraType?, curHP?, boostedStat?}
 *          (teraType set = terastallized, as in @smogon/calc)
 *       M: {name, isCrit?, hits?}   F: @smogon/calc Field options (weather, terrain, attackerSide, defenderSide)
 *       -> {"damage": [16 rolls] | number, "desc": string}
 *   {"cmd": "calcs", "cases": [{attacker, defender, move, field}, ...]} -> {"results": [{damage, desc} | {error}]}
 *   {"cmd": "start", "format": "gen9ou", "p1": packed, "p2": packed, "seed"?: [4 ints], "forced"?: FORCED}
 *       -> {"battle": n, "state": STATE}
 *   {"cmd": "choose", "battle": n, "p1"?: choice, "p2"?: choice}
 *       Showdown choice strings ("team 213456", "move 1", "move 2 terastallize", "switch 3", "default").
 *       -> {"state": STATE} after both sides' choices are in (or the error message for a bad choice)
 *   {"cmd": "end", "battle": n}
 *
 * FORCED (forced-outcome mode; every field optional, defaults shown) replaces Showdown's random draws
 * by fixed outcomes. It overrides the battle object's random hooks and wraps the battle-actions
 * methods that make each kind of draw, so Showdown's source is untouched:
 *   hit: true            accuracy checks succeed
 *   crit: false          critical hits
 *   roll: 0              damage roll index (0 = max damage, 15 = min)
 *   secondaries: false   secondary / self-drop chances (e.g. Scald burn, Draco Meteor always drops)
 *   fullPara: false      paralysis full-paralysis chance
 *   thaw: false          freeze thaw chance
 *   confusionSelfHit: false
 *   stall: true          Protect-family success on consecutive use
 *   other: true          any other randomChance()
 *   randomMin: true      random(m, n) for durations etc. returns m (e.g. sleep lasts its minimum)
 * Speed ties keep a fixed order (no shuffle: the earlier-listed action, i.e. p1, goes first).
 *
 * STATE: {turn, ended, winner, field: {weather, weatherTurns, terrain, terrainTurns, pseudoWeather},
 *         sides: [{id, active: [slot], sideConditions, pokemon: [MON]}], requests: {p1, p2}, log: [new lines]}
 *   MON: {name, species, slot, hp, maxhp, status, statusTurns, toxicStage, boosts, item, ability, teraType,
 *         terastallized, volatiles, moves: [{id, pp, maxpp, disabled}], fainted, active}
 */
'use strict';

const path = require('path');
const readline = require('readline');

const showdownDir = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', '..', 'pokemon-showdown'));
const sim = require(path.join(showdownDir, 'dist', 'sim'));
const {Battle, Teams, Dex} = sim;
const calc = require('@smogon/calc');

const CALC_GEN = calc.Generations.get(9);
const battles = new Map();
let nextBattle = 1;

// --- forced-outcome mode -----------------------------------------------------------------------

const FORCED_DEFAULTS = {
	hit: true, crit: false, roll: 0, secondaries: false, fullPara: false, thaw: false,
	confusionSelfHit: false, stall: true, other: true, randomMin: true,
};

function installForced(battle, opts) {
	const o = {...FORCED_DEFAULTS, ...opts};
	const ctx = [];
	const wrap = (obj, name, tag) => {
		const orig = obj[name];
		obj[name] = function (...args) {
			ctx.push(tag);
			try {
				return orig.apply(this, args);
			} finally {
				ctx.pop();
			}
		};
	};
	wrap(battle.actions, 'hitStepAccuracy', 'accuracy');
	wrap(battle.actions, 'hitStepMoveHitLoop', 'hitloop');
	wrap(battle.actions, 'getDamage', 'crit');
	wrap(battle, 'randomizer', 'roll');
	wrap(battle.actions, 'secondaries', 'secondary');
	wrap(battle.actions, 'selfDrops', 'secondary');
	const top = () => ctx[ctx.length - 1];

	battle.randomChance = (num, den) => {
		const t = top();
		if (t === 'accuracy' || (t === 'hitloop' && den === 100)) return o.hit;
		if (t === 'crit') return o.crit;
		if (num === 1 && den === 4) return o.fullPara;
		if (num === 1 && den === 5) return o.thaw;
		if (num === 33 && den === 100) return o.confusionSelfHit;
		if (num === 1) return o.stall; // Protect stall counter: randomChance(1, counter)
		return o.other;
	};
	battle.random = (m, n) => {
		const t = top();
		if (t === 'roll') return o.roll;
		if (t === 'secondary') return o.secondaries ? 0 : 99;
		if (n === undefined) return m === undefined ? 0 : (o.randomMin ? 0 : m - 1);
		return o.randomMin ? m : n - 1;
	};
	battle.sample = items => items[0];
	battle.prng.shuffle = () => {};
}

// --- state snapshots ---------------------------------------------------------------------------

function monState(p, slot) {
	return {
		name: p.name,
		species: p.species.id,
		slot,
		hp: p.hp,
		maxhp: p.maxhp,
		status: p.status || '',
		statusTurns: p.statusState && p.statusState.time !== undefined ? p.statusState.time : null,
		toxicStage: p.status === 'tox' ? p.statusState.stage : 0,
		boosts: {...p.boosts},
		item: p.item || '',
		ability: p.ability || '',
		teraType: p.teraType,
		terastallized: p.terastallized || '',
		volatiles: Object.keys(p.volatiles).sort(),
		moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled})),
		fainted: !!p.fainted,
		active: !!p.isActive,
		stats: {...p.storedStats},
	};
}

function sideState(side) {
	const conds = {};
	for (const [id, s] of Object.entries(side.sideConditions)) {
		conds[id] = {layers: s.layers || 1, duration: s.duration === undefined ? null : s.duration};
	}
	return {
		id: side.id,
		active: side.active.map(p => (p ? side.pokemon.indexOf(p) : null)),
		sideConditions: conds,
		pokemon: side.pokemon.map((p, i) => monState(p, i)),
	};
}

function battleState(b, logFrom) {
	const f = b.field;
	return {
		turn: b.turn,
		ended: !!b.ended,
		winner: b.winner || '',
		field: {
			weather: f.weather || '',
			weatherTurns: f.weather ? f.weatherState.duration : null,
			terrain: f.terrain || '',
			terrainTurns: f.terrain ? f.terrainState.duration : null,
			pseudoWeather: Object.fromEntries(
				Object.entries(f.pseudoWeather).map(([k, s]) => [k, s.duration === undefined ? null : s.duration])
			),
		},
		sides: b.sides.map(sideState),
		requests: {p1: b.sides[0].activeRequest || null, p2: b.sides[1].activeRequest || null},
		log: b.log.slice(logFrom),
	};
}

// --- commands ----------------------------------------------------------------------------------

function toPacked(team) {
	if (!team.includes('\n') && team.includes('|')) return team;
	return Teams.pack(Teams.import(team));
}

function cmdStats(req) {
	return {
		stats: req.teams.map(team => {
			const b = new Battle({formatid: 'gen9customgame'});
			b.setPlayer('p1', {name: 'p1', team: toPacked(team)});
			b.setPlayer('p2', {name: 'p2', team: toPacked(team)});
			return b.sides[0].pokemon.map(p => ({
				species: p.species.id,
				maxhp: p.maxhp,
				stats: {...p.storedStats},
			}));
		}),
	};
}

function calcPokemon(p) {
	const opts = {};
	for (const k of ['level', 'item', 'ability', 'nature', 'evs', 'ivs', 'boosts', 'status', 'teraType', 'curHP', 'boostedStat']) {
		if (p[k] !== undefined && p[k] !== null && p[k] !== '') opts[k] = p[k];
	}
	return new calc.Pokemon(CALC_GEN, p.species, opts);
}

function cmdCalc(req) {
	const result = calc.calculate(
		CALC_GEN,
		calcPokemon(req.attacker),
		calcPokemon(req.defender),
		new calc.Move(CALC_GEN, req.move.name, {isCrit: !!req.move.isCrit, hits: req.move.hits}),
		new calc.Field(req.field || {})
	);
	let desc = '';
	try {
		desc = result.desc();
	} catch (e) {}
	return {damage: result.damage, desc};
}

function cmdStart(req) {
	const opts = {formatid: req.format || 'gen9ou'};
	if (req.seed) opts.seed = req.seed;
	const b = new Battle(opts);
	if (req.forced) installForced(b, req.forced === true ? {} : req.forced);
	b.setPlayer('p1', {name: 'p1', team: toPacked(req.p1)});
	b.setPlayer('p2', {name: 'p2', team: toPacked(req.p2)});
	const id = nextBattle++;
	battles.set(id, {b, logPos: 0});
	const entry = battles.get(id);
	const state = battleState(b, 0);
	entry.logPos = b.log.length;
	return {battle: id, state};
}

function cmdChoose(req) {
	const entry = battles.get(req.battle);
	if (!entry) throw new Error(`no battle ${req.battle}`);
	const b = entry.b;
	for (const side of ['p1', 'p2']) {
		if (req[side] === undefined || req[side] === null) continue;
		if (!b.choose(side, req[side])) {
			const err = b.sides[side === 'p1' ? 0 : 1].choice.error || 'invalid choice';
			throw new Error(`${side}: ${err}`);
		}
	}
	const state = battleState(b, entry.logPos);
	entry.logPos = b.log.length;
	return {state};
}

const COMMANDS = {
	pack: req => ({packed: req.teams.map(toPacked)}),
	stats: cmdStats,
	calc: cmdCalc,
	calcs: req => ({
		results: req.cases.map(c => {
			try {
				return cmdCalc(c);
			} catch (e) {
				return {error: String(e && e.message || e)};
			}
		}),
	}),
	start: cmdStart,
	choose: cmdChoose,
	end: req => (battles.delete(req.battle), {}),
	ping: () => ({showdownCommit: null}),
};

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
rl.on('line', line => {
	if (!line.trim()) return;
	let req;
	try {
		req = JSON.parse(line);
		const fn = COMMANDS[req.cmd];
		if (!fn) throw new Error(`unknown cmd ${req.cmd}`);
		process.stdout.write(JSON.stringify({id: req.id, ok: true, ...fn(req)}) + '\n');
	} catch (e) {
		process.stdout.write(JSON.stringify({id: req && req.id, ok: false, error: String(e && e.message || e)}) + '\n');
	}
});
