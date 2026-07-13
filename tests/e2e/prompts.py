"""The 20 fixed plain-text prompts for the end-to-end HF parity suite (§12c).

These are *plain-text continuations*, not chat turns: the parity harness
tokenizes them WITHOUT the chat template (``add_special_tokens=False``) so the
Redline engine and HF transformers consume byte-identical token IDs. Every
prompt trails off mid-thought so greedy continuation stays on-topic and is
unlikely to emit an end-of-text token inside the first 128 generated tokens.

Composition (§12c):
  - 5 short  prompts   (nominal 50-100 prompt tokens)
  - 10 medium prompts  (nominal 100-300 prompt tokens)
  - 5 long   prompts   (nominal 300-500 prompt tokens)

The token bands are *nominal*: exact counts depend on the Qwen2 byte-level BPE
tokenizer, so ``gen_reference.py`` records the real per-prompt token count and
warns (never fails) when a prompt drifts outside its band. This module is the
single source of truth for the prompt text; ``gen_reference.py`` imports it,
tokenizes, and embeds the resulting IDs in the reference JSON, and
``test_hf_parity.py`` cross-checks the reference against this list so a stale
reference is caught before any parity comparison runs.

The prompts are neutral, factual, expository English chosen to continue
coherently; none contains chat markers or special tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """One fixed parity prompt."""

    index: int
    category: str  # "short" | "medium" | "long"
    text: str


# Nominal prompt-token bands per category (§12c). Used only for advisory
# warnings in gen_reference.py; nothing gates on them.
CATEGORY_TOKEN_BANDS: dict[str, tuple[int, int]] = {
    "short": (50, 100),
    "medium": (100, 300),
    "long": (300, 500),
}


_SHORT: list[str] = [
    # 0
    "The Pacific Ocean is the largest and deepest of Earth's five oceans, "
    "stretching from the Arctic in the north all the way to the Southern Ocean "
    "and covering more than sixty million square miles. Its waters wash against "
    "the shores of Asia, Australia, and the Americas, and far beneath its "
    "restless surface lie",
    # 1
    "Photosynthesis is the process by which green plants, algae, and certain "
    "bacteria convert light energy into the chemical energy stored in sugars. "
    "Using sunlight captured by chlorophyll, water drawn up through the roots, "
    "and carbon dioxide absorbed from the surrounding air, these organisms build "
    "glucose and release oxygen, and the very first step in this remarkable "
    "sequence is",
    # 2
    "The printing press that Johannes Gutenberg assembled in Europe around the "
    "middle of the fifteenth century transformed the way knowledge was recorded "
    "and shared. Before it, books were copied slowly by hand and remained rare "
    "and costly, but afterward ideas could travel quickly across the continent, "
    "and one of the earliest and most far-reaching consequences was",
    # 3
    "Honeybees live in tightly organized colonies that may contain tens of "
    "thousands of individuals, each with a defined role to perform. The single "
    "queen lays the eggs, the workers forage for nectar and tend the comb, and "
    "the drones exist only to mate, yet the survival of the whole colony through "
    "a long winter depends above all on",
    # 4
    "Mount Everest, rising on the border between Nepal and the Tibet Autonomous "
    "Region of China, is the highest mountain above sea level anywhere on Earth. "
    "Climbers who set out to reach its distant summit must contend with thinning "
    "air, sudden violent storms, and hidden crevasses, and the single greatest "
    "danger that experienced mountaineers describe is",
]


_MEDIUM: list[str] = [
    # 5
    "The water cycle describes the continuous movement of water on, above, and "
    "below the surface of the Earth, and it is one of the most important "
    "processes sustaining life on the planet. Warmed by the sun, water "
    "evaporates from oceans, lakes, and rivers, rising as invisible vapor into "
    "the atmosphere. As that vapor climbs and cools, it condenses into the tiny "
    "droplets that gather to form clouds. When those droplets grow heavy enough, "
    "they fall back to the ground as rain, snow, sleet, or hail, a stage that "
    "scientists call precipitation. Some of the water that reaches the ground "
    "soaks downward and is stored underground, some is taken up by plants, and "
    "the rest runs across the land in streams that eventually return it to the "
    "sea. What makes the cycle so remarkable, and the reason it never runs out "
    "of energy to keep turning, is",
    # 6
    "At the height of its power the Roman Empire was bound together by an "
    "extraordinary network of roads that reached from the misty frontiers of "
    "Britain to the deserts of Syria and North Africa. These roads were built to "
    "last: engineers dug deep foundations, layered crushed stone and gravel for "
    "drainage, and topped the surface with tightly fitted paving blocks that "
    "shed rain and carried heavy traffic for centuries. Along them marched the "
    "legions that defended the empire, and behind the legions came merchants, "
    "messengers, tax collectors, and ordinary travelers. Milestones marked the "
    "distances, and way stations offered fresh horses and shelter. The roads "
    "knit distant provinces into a single economy and a single administration, "
    "and the phrase that all roads lead to Rome captured a literal truth, "
    "because the deeper purpose these highways served was",
    # 7
    "The human circulatory system is a closed network of vessels that carries "
    "blood to every living cell in the body, and at its center sits the heart, a "
    "muscular pump roughly the size of a clenched fist. With each beat the heart "
    "drives oxygen-rich blood out through the arteries, which branch again and "
    "again into ever smaller vessels until they become capillaries so narrow "
    "that red cells pass through in single file. In those thin-walled capillaries "
    "oxygen and nutrients slip out to the tissues while carbon dioxide and waste "
    "move in, and the now oxygen-poor blood begins its return journey through the "
    "veins back toward the heart and lungs. The whole circuit repeats itself "
    "thousands of times a day without pause, and the feature of this design that "
    "keeps the blood flowing in one direction rather than sloshing backward is",
    # 8
    "The origins of the internet reach back to the late 1960s, when researchers "
    "funded by the United States Department of Defense built an experimental "
    "network called ARPANET to connect a handful of university computers. Their "
    "goal was to let expensive, geographically scattered machines share resources "
    "and to keep communicating even if part of the network failed. The key idea "
    "that made this possible was packet switching, in which messages are broken "
    "into small labeled pieces that travel independently and are reassembled at "
    "their destination. Over the following decades new common languages, or "
    "protocols, allowed separate networks to interconnect into one global network "
    "of networks, and the invention of the World Wide Web layered an easy system "
    "of linked pages on top of it. The single development that finally carried "
    "the internet from research laboratories into ordinary homes was",
    # 9
    "Volcanoes are among the most dramatic expressions of the restless activity "
    "taking place deep inside the Earth. The planet's rigid outer shell is broken "
    "into enormous slabs called tectonic plates that drift slowly over a hot, "
    "partly molten layer beneath them. Where two plates pull apart or one grinds "
    "beneath another, rock can melt into magma, a mixture of liquid rock, "
    "dissolved gases, and crystals that is lighter than the solid rock around it. "
    "That buoyant magma rises through cracks and collects in chambers, and when "
    "the pressure grows too great it forces its way to the surface in an "
    "eruption. Some eruptions ooze gently as rivers of glowing lava, while others "
    "explode with terrifying force, hurling ash miles into the sky. What actually "
    "decides whether a given volcano erupts quietly or violently is",
    # 10
    "The moon exerts a gentle but relentless gravitational pull on the Earth, and "
    "the most visible result of that pull is the rise and fall of the tides. As "
    "the Earth turns beneath the moon, the moon's gravity tugs hardest on the "
    "ocean water directly below it, drawing it into a bulge, while a second bulge "
    "forms on the opposite side of the planet. Coastlines passing through these "
    "bulges experience high tide, and the shores between them experience low "
    "tide, so most places see roughly two high tides and two low tides each day. "
    "The sun adds its own weaker tidal pull, and when the sun, moon, and Earth "
    "line up, the tides swing to their greatest extremes. The reason a bulge "
    "appears on the far side of the Earth, where the moon's pull is actually "
    "weakest, is",
    # 11
    "Coffee began as a wild shrub in the highlands of northeastern Africa, and "
    "the story of how its bitter roasted seeds became one of the most traded "
    "commodities on Earth spans many centuries. According to a much-repeated "
    "legend, a herder noticed his goats growing lively after nibbling the bright "
    "red cherries of a certain bush. Whatever its true beginnings, the drink "
    "spread first through the Arabian Peninsula, where it was brewed in homes and "
    "in the public coffeehouses that became lively centers of conversation and "
    "trade. From there merchants and travelers carried the beans across the "
    "Mediterranean and into Europe, and later colonial plantations pushed "
    "cultivation into the Americas and Asia. Today coffee links millions of small "
    "farmers to distant drinkers through a long chain of hands, and the step in "
    "that journey that transforms a green, almost flavorless seed into the "
    "fragrant brown bean we recognize is",
    # 12
    "Electricity was a mysterious curiosity for most of human history, studied in "
    "the form of crackling sparks and the shock of certain fish, long before "
    "anyone could put it to work. The decisive advances came in the nineteenth "
    "century, when experimenters learned that electricity and magnetism are "
    "intimately linked. Michael Faraday showed that moving a magnet near a coil "
    "of wire causes an electric current to flow in the wire, a phenomenon called "
    "electromagnetic induction. That single discovery is the principle behind the "
    "generators that produce nearly all of the world's electric power, from vast "
    "hydroelectric dams to the alternator turning in a car engine. It also "
    "underlies the electric motor, which runs the same process in reverse to turn "
    "electrical energy back into motion. The reason induction produces a current "
    "only while the magnet is actually moving, and not while it sits still, is",
    # 13
    "Tropical rainforests cover only a small fraction of the Earth's land "
    "surface, yet they are home to more than half of all the plant and animal "
    "species known to science. Warm temperatures and heavy rainfall throughout "
    "the year create conditions in which life grows in dense, competing layers. "
    "High above, the crowns of the tallest trees form a nearly continuous canopy "
    "that captures most of the sunlight; below it lies a shadowy understory of "
    "smaller trees and vines, and on the forest floor, where little light "
    "reaches, fungi and insects break down fallen leaves with astonishing speed. "
    "Because the plants recycle nutrients so quickly, the soil itself is often "
    "surprisingly poor. Countless species have evolved narrow, specialized ways "
    "of living that exist nowhere else, and the main reason such an enormous "
    "variety of life can crowd into so small an area is",
    # 14
    "The steam engine was the machine that powered the Industrial Revolution and "
    "reshaped the modern world. Early versions were built to pump water out of "
    "deep mines, where flooding constantly threatened to halt the digging. These "
    "first engines were slow and wasteful, burning huge quantities of coal to do "
    "modest amounts of work. The great improvement came when James Watt added a "
    "separate condenser, so the main cylinder no longer had to be heated and "
    "cooled with every stroke, dramatically raising efficiency. A practical, "
    "reasonably economical engine could now be built almost anywhere, not just "
    "beside a coal pit, and it was soon harnessed to drive factory machinery, "
    "locomotives, and steamships. Work that had always depended on muscle, wind, "
    "or falling water could suddenly be done on demand, and the deepest change "
    "this new source of reliable power set in motion was",
]


_LONG: list[str] = [
    # 15
    "The French Revolution, which began in 1789, was one of the great turning "
    "points in the history of the modern world, and its causes had been building "
    "for many years. France was governed by an absolute monarchy and divided into "
    "rigid legal orders: a small, privileged nobility and clergy who paid little "
    "tax, and a vast common population, the Third Estate, that carried nearly the "
    "entire burden of the state. Years of costly wars, an extravagant royal "
    "court, and poor harvests had pushed the government toward bankruptcy while "
    "ordinary people struggled to afford bread. Enlightenment writers had spent "
    "decades arguing that political authority should rest on reason and the "
    "consent of the governed rather than on the accident of birth, and their "
    "ideas circulated widely in pamphlets and salons. When the king summoned a "
    "long-dormant assembly to approve new taxes, the representatives of the Third "
    "Estate refused to be outvoted by the privileged orders and declared "
    "themselves a National Assembly with the authority to write a constitution. "
    "The storming of a royal fortress in Paris turned a political crisis into a "
    "popular uprising, and within a few years an ancient monarchy had been swept "
    "away and replaced by a republic. The new government abolished the feudal "
    "privileges of the old order, issued a sweeping declaration of rights, and "
    "redrew the administration of the entire country, but it also slid into war "
    "with its neighbors and turned violently on its own citizens during the "
    "period remembered as the Terror. Historians still debate exactly why the "
    "revolution spiraled from cautious reform into radical upheaval and terror, "
    "but the single factor most of them point to first is",
    # 16
    "The theory of evolution by natural selection, set out by Charles Darwin in "
    "the middle of the nineteenth century, is the central organizing idea of all "
    "modern biology. Darwin began from a few plain observations. Living things "
    "produce far more offspring than can possibly survive, so in every generation "
    "there is a struggle for limited food, space, and mates. The individuals "
    "within any population are not identical; they vary in countless small ways, "
    "and much of that variation can be inherited by their descendants. From these "
    "facts a powerful conclusion follows almost inevitably: individuals whose "
    "particular traits happen to suit them better to their surroundings tend, on "
    "average, to survive longer and leave more offspring, so those advantageous "
    "traits become more common in the population over time. Given enough "
    "generations, this slow, unguided filtering can reshape a lineage completely "
    "and, when populations are separated and adapt to different conditions, split "
    "one ancestral species into many. Darwin marshaled evidence from many "
    "quarters — the graded beaks of island finches, the bones shared by seemingly "
    "unrelated animals, and the way breeders had reshaped pigeons and dogs in mere "
    "decades — to argue that the same patient process had been at work in wild "
    "nature across almost unimaginable spans of time. The idea provoked fierce "
    "controversy when it was published, yet within a generation most working "
    "naturalists had accepted that species change over time. Darwin lacked any "
    "knowledge of genes and could not explain how traits were passed on or where "
    "new variation came from, a gap that troubled him deeply. The later discovery "
    "that answered those questions and placed his theory on a secure mechanical "
    "foundation was",
    # 17
    "The modern electronic computer did not appear all at once but emerged from "
    "the work of many people wrestling with the same basic problem: how to carry "
    "out long, error-prone calculations quickly and reliably. Early mechanical "
    "calculators could add and multiply, and in the nineteenth century Charles "
    "Babbage designed, though he never finished, an astonishing machine meant to "
    "be programmed with punched cards. The decisive breakthroughs came in the "
    "middle of the twentieth century. Engineers first built enormous machines "
    "filled with thousands of glowing vacuum tubes that could be rewired or, "
    "later, programmed to run different tasks, but the tubes were bulky, hot, and "
    "prone to failure. A single burned-out tube among thousands could halt a "
    "calculation, and technicians walked the aisles replacing them by hand. "
    "Alongside the hardware came an equally important idea: that "
    "the list of instructions a machine follows could itself be stored in the same "
    "memory as the data it works on, so a single physical device could be turned to "
    "countless different tasks simply by loading a new program into it. The "
    "invention of the transistor, a tiny solid-state switch "
    "with no fragile filament, changed everything, and the later ability to etch "
    "many transistors together onto a single chip of silicon set off a steady "
    "shrinking and cheapening of computing power that has continued for decades. "
    "A machine that once filled a room and cost a fortune now fits in a pocket "
    "and answers to a fingertip. Underlying every one of these machines, from the "
    "room-sized pioneers to the phone in your hand, is a single simple idea about "
    "how information can be represented and manipulated, and that foundational "
    "idea is",
    # 18
    "Our solar system formed about four and a half billion years ago from an "
    "enormous, slowly rotating cloud of gas and dust drifting through space. "
    "Something, perhaps the shock wave from a nearby exploding star, disturbed "
    "the cloud and caused a dense region to begin collapsing under its own "
    "gravity. As the cloud shrank, it spun faster and flattened into a broad "
    "disk, with most of the material piling up at the center. There the pressure "
    "and temperature climbed until nuclear fusion ignited, and the sun was born, "
    "a star holding more than ninety-nine percent of all the mass in the system. "
    "In the leftover disk swirling around the young sun, tiny grains collided and "
    "stuck together, growing over long ages into pebbles, boulders, and finally "
    "worlds. Close to the sun, where it was too hot for ices to survive, only "
    "rock and metal could condense, and these built the small, dense inner "
    "planets, including the Earth. Farther out, beyond a boundary sometimes "
    "called the frost line, water and other substances froze solid, giving the "
    "outer planets far more material to gather, so they swelled into giants "
    "wrapped in deep atmospheres of gas. Not everything was swept up into a planet; "
    "belts of rocky rubble and countless icy bodies were left orbiting in the gaps "
    "and in the cold outer reaches, and the largest of the giant planets shepherded "
    "and scattered much of this leftover debris with its powerful gravity. The "
    "clearest reason the inner planets "
    "and the outer planets ended up so profoundly different from one another is",
    # 19
    "The Silk Road was never a single paved highway but a shifting web of trade "
    "routes that linked the civilizations of East Asia, Central Asia, the Middle "
    "East, and the Mediterranean world for well over a thousand years. Its name, "
    "coined only much later, comes from the fine Chinese silk that was carried "
    "westward and prized as a luxury in distant markets, but silk was only one of "
    "countless goods that moved along these tracks. Caravans of camels crossed "
    "deserts and mountain passes bearing spices, precious stones, glassware, "
    "paper, medicines, and metals, while few merchants ever traveled the whole "
    "distance themselves; instead goods passed from hand to hand through a long "
    "chain of intermediaries, each taking a share of the profit. Great caravan "
    "cities such as Samarkand and the oasis town of Dunhuang flourished as meeting "
    "points where traders from many nations bargained, worshipped, and exchanged "
    "news, and rulers competed to guard the routes because the tolls and taxes they "
    "yielded were a source of enormous wealth. Along with cargo "
    "traveled things that no ledger recorded. Religions, among them Buddhism, "
    "spread along the routes into new lands; artistic styles, scientific "
    "knowledge, and technologies such as papermaking diffused across the "
    "continents; and languages mingled in the bustling oasis towns that grew rich "
    "serving the trade. Unfortunately, disease traveled the same paths, and "
    "outbreaks of plague followed the caravans and ships. Considering everything "
    "that flowed along these routes over the centuries, the most lasting "
    "importance of the Silk Road was probably not the trade goods themselves but",
]


def _build() -> list[Prompt]:
    prompts: list[Prompt] = []
    idx = 0
    for text in _SHORT:
        prompts.append(Prompt(index=idx, category="short", text=text))
        idx += 1
    for text in _MEDIUM:
        prompts.append(Prompt(index=idx, category="medium", text=text))
        idx += 1
    for text in _LONG:
        prompts.append(Prompt(index=idx, category="long", text=text))
        idx += 1
    return prompts


PROMPTS: list[Prompt] = _build()

# Fail loudly at import if the composition ever drifts from the §12c contract.
assert len(PROMPTS) == 20, f"expected 20 prompts, got {len(PROMPTS)}"
assert sum(p.category == "short" for p in PROMPTS) == 5
assert sum(p.category == "medium" for p in PROMPTS) == 10
assert sum(p.category == "long" for p in PROMPTS) == 5


def all_prompts() -> list[Prompt]:
    """Return the 20 fixed prompts in canonical order (short, medium, long)."""
    return list(PROMPTS)
