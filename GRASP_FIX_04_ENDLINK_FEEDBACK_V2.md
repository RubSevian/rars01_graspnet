# Fix 04 — фактический приход End_link по feedback

## Цель

Перед переходом к следующему этапу убедиться, что реальный `End_link` действительно достиг целевой Cartesian-позы по свежему feedback.

## 1. Проверка фактической позы

Использовать измеренные углы суставов руки:

$$
q_{measured}
$$

и вычислять:

$$
T^{actual}_{End}=FK(q_{measured})
$$

Позиционная ошибка:

$$
e_p=\|p^{actual}_{End}-p^{target}_{End}\|
$$

Ошибка ориентации:

$$
e_R=
\arccos\left(
\operatorname{clip}
\left(
\frac{\operatorname{tr}(R_t^TR_a)-1}{2},
-1,1
\right)
\right)
$$

Для FK использовать ту же модель, `End_link` и систему углов, что используются в IK.

Мотор гриппера из проверки суставов руки исключить.

## 2. Допуски

Согласовать Fix 02 и Fix 04.

### Fix 02 — планируемая конечная поза

Для финального grasp использовать примерно:

```yaml
planning_fk:
  position_tolerance_m: 0.004
  orientation_tolerance_deg: 2.0
```

### Fix 04 — реальное исполнение

```yaml
motion_feedback:
  grasp_position_tolerance_m: 0.005
  grasp_orientation_tolerance_deg: 3.0
  velocity_tolerance_deg_s: 2.0
  settle_time_s: 0.2
  timeout_margin_s: 1.5
```

Для `pregrasp` можно задать отдельные немного более мягкие допуски.

## 3. Свежесть feedback

Нельзя засчитывать один и тот же cached state как стабильное состояние в течение `0.2 s`.

Каждый feedback должен быть новым/свежим для всех суставов руки.

Если:

- feedback устарел;
- measurement timestamp/sequence не обновился;
- `raw.valid == false`;
- получен `NaN`;
- transport error;

то:

- сбросить `settle timer`;
- при критической ошибке выполнить `ABORT`.

SDK не менять без необходимости. Сначала использовать доступный timestamp / sequence / update marker существующего feedback.

## 4. Условие READY

Следующий этап разрешён только если одновременно:

1. отправка текущей траектории завершена;
2. feedback свежий;
3. Cartesian ошибки находятся в допуске;
4. скорость всех суставов руки ниже порога;
5. условия непрерывно выполняются не менее `settle_time_s`.

Если хотя бы одно условие нарушилось во время выдержки — `settle timer` сбрасывается.

## 5. Deadline

Использовать один deadline от начала движения:

$$
t_{deadline}
=
t_{start}
+
T_{planned}
+
T_{margin}
$$

где `T_planned` — фактически рассчитанная длительность исполняемой траектории с учётом ограничений скорости и периода отправки команд.

Не ждать сначала завершения движения, а затем ещё один полный timeout.

## 6. Поведение при отказе

### Ошибка `pregrasp`

```text
ABORT
не начинать Cartesian approach
не закрывать gripper
```

### Ошибка финального `grasp`

```text
ABORT
не закрывать gripper
```

Не продолжать последовательность автоматически.

## 7. Логирование

Сохранять:

```text
planned duration
feedback age / sequence
q_measured
q_target
Cartesian position error
Cartesian orientation error
max arm joint velocity
settle timer
success / timeout / stale feedback / transport error
```

## Итоговая логика

```text
trajectory start
→ fresh feedback
→ trajectory sending finished
→ End_link inside Cartesian tolerance
→ low joint velocity
→ stable for 0.2 s
→ READY
→ next stage / close gripper
```

Fix 02 проверяет план:

$$
T_{target}\rightarrow IK\rightarrow q_{target}\rightarrow FK
$$

Fix 04 проверяет реальное исполнение:

$$
q_{measured}\rightarrow FK\rightarrow T^{actual}_{End}
$$
