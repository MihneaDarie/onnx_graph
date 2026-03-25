#[macro_export]
macro_rules! call_maxpool_for_typed_array {
    ($self:expr, $kernel:expr, $strides:expr, $pads:expr, $dilations:expr, $ceil_mode:expr, $o:expr, [$($variant:ident),+]) => {
        match $self {
            $(
                TypedArray::$variant(x) => {
                    max_pool_variant!($variant, $kernel, $strides, $pads, $dilations, $ceil_mode, x, $o)
                }
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for max_pool")),
        }
    };
}

#[macro_export]
macro_rules! max_pool_variant {
    ($variant:ident, $kernel:expr, $strides:expr, $pads:expr, $dilations:expr, $ceil_mode:expr, $x:expr, $o:expr) => {{
        let x4 = $x.view().into_dimensionality::<Ix4>()?;
        let (batch, channels, hin, win) = x4.dim();

        let kh = $kernel[0];
        let kw = $kernel[1];
        let sh = $strides.first().copied().unwrap_or(1);
        let sw = $strides.get(1).copied().unwrap_or(1);
        let ph = $pads.first().copied().unwrap_or(0);
        let pw = $pads.get(1).copied().unwrap_or(0);
        let dh = $dilations.first().copied().unwrap_or(1);
        let dw = $dilations.get(1).copied().unwrap_or(1);

        let hout = if $ceil_mode != false {
            (hin + 2 * ph - dh * (kh - 1) - 1 + sh - 1) / sh + 1
        } else {
            (hin + 2 * ph - dh * (kh - 1) - 1) / sh + 1
        };
        let wout = if $ceil_mode != false {
            (win + 2 * pw - dw * (kw - 1) - 1 + sw - 1) / sw + 1
        } else {
            (win + 2 * pw - dw * (kw - 1) - 1) / sw + 1
        };

        let mut out = ArrayD::from_elem(
            IxDyn(&[batch, channels, hout, wout]),
            $x.iter().next().copied().unwrap(),
        );

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut max_val = None;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * sh + khi * dh;
                                let iw = ow * sw + kwi * dw;
                                if ih >= ph && iw >= pw {
                                    let ih = ih - ph;
                                    let iw = iw - pw;
                                    if ih < hin && iw < win {
                                        let val = x4[[b, c, ih, iw]];
                                        max_val = Some(match max_val {
                                            None => val,
                                            Some(m) => {
                                                if val > m {
                                                    val
                                                } else {
                                                    m
                                                }
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        out[[b, c, oh, ow]] = max_val.unwrap();
                    }
                }
            }
        }
        *($o) = TypedArray::$variant(out);
    }};
}
